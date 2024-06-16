import os, sys, time
import importlib
import torch
from social_trans import SocialTrans
from data import Dataloader
from utils import seed, get_rng_state
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs='+', default=['data/zara02/train'])
parser.add_argument("--test", nargs='+', default=['data/zara02/test'])
parser.add_argument("--frameskip", type=int, default=1)
parser.add_argument("--config", type=str, default='config/zara02.py')
parser.add_argument("--ckpt", type=str, default='models/zara02')
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)


if __name__ == "__main__":
        settings = parser.parse_args()
        print('settings.config',settings.config)
        spec = importlib.util.spec_from_file_location("config", settings.config)   
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)  

        if settings.device is None:
                settings.device = "cuda" if torch.cuda.is_available() else "cpu"
        settings.device = torch.device(settings.device)

        seed(settings.seed)
        init_rng_state = get_rng_state(settings.device)
        rng_state = init_rng_state

        ###############################################################################
        #####                                                                    ######
        ##### prepare datasets                                                   ######
        #####                                                                    ######
        ###############################################################################
        kwargs = dict(
                batch_first=False, frameskip=settings.frameskip,
                ob_horizon=config.OB_HORIZON, pred_horizon=config.PRED_HORIZON,
                device=settings.device, seed=settings.seed)
        # 通过关键字dict和关键字参数创建

        if settings.test:
                if config.INCLUSIVE_GROUPS is not None:
                        inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.test))]    # [[]]
                else:
                        inclusive = None
                
                test_dataset = Dataloader(
                        settings.test, **kwargs, inclusive_groups=inclusive,
                        batch_size=config.BATCH_SIZE, shuffle=False
                )
                test_data = torch.utils.data.DataLoader(test_dataset, 
                collate_fn=test_dataset.collate_fn,
                batch_sampler=test_dataset.batch_sampler
                )

        ztrans = SocialTrans(4,3,config.OB_HORIZON,config.PRED_HORIZON).to(settings.device)
        ztrans.load_state_dict(torch.load('put_pre-trained_model_here.pth')['model'])
        ztrans.eval()

        fde = []
        with torch.no_grad():
                num = 0
                ade = 0
                fde = 0
                for x, y, nei in test_data:
                        pred = ztrans(x,nei[:config.PRED_HORIZON])
                        if len(pred.size()) == 4:
                                losstrans = ztrans.loss(pred, y.unsqueeze(0).repeat(config.PRED_SAMPLES,1,1,1))
                        
                        else:
                                losstrans = ztrans.loss(pred, y)
                        x = x.cpu().numpy()
                        y = y.cpu().numpy()
                        nei = nei.cpu().numpy()
                        pred = pred.cpu().numpy()

                        num = num + 1
                        ade = ade + losstrans['ade'].item()
                        fde = fde + losstrans['fde'].item()

                print('-----------------------------------------')
                print('ade',ade/num,'fde',fde/num)
