from tqdm import tqdm
import os
import argparse
import collections
import torch
import torch.nn as nn
from datasets.CBAS_Testset import CBASDataset
from models.DCNet import  dcnet_vit_tiny, dcnet_vit_small, dcnet_vit_base
from PIL import Image
from torch.autograd import Variable


def get_dataset(opts):
    val_dst = CBASDataset(is_train=False,voc_dir=opts.testset_path)
    return val_dst


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')

    p8 = im.convert("P")  # 将24位深的RGB图像转化为8位深的模式“P”图像
    p8.save(d_dir+image_name+'.png')


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--prediction_dir", type=str, 
        default='../CBAS_Preds/',
        help="path to Dataset")
    
    parser.add_argument("--testset_path", type=str, 
        default='../Dataset/CBAS/Test',
        help="path to Dataset")
    
    parser.add_argument("--dataset", type=str, default='CBAS', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")

    parser.add_argument("--model", type=str, default='dcnet_vit_small',
        help='model name:[dcnet_vit_small]')

    parser.add_argument("--batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--trainsize", type=int, default=512)
    parser.add_argument("--n_cpu", type=int, default=1,
                        help="download datasets")

    parser.add_argument("--ckpt",type=str,
                default=None, 
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser


def main():
  
    opts = get_argparser().parse_args()

    test_dst = get_dataset(opts)

    test_loader = torch.utils.data.DataLoader(
        test_dst, batch_size=opts.batch_size, shuffle=False, num_workers=opts.n_cpu)
    print("Dataset: %s, Test set: %d" %
            (opts.dataset, len(test_dst)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = eval(opts.model)(num_classes=opts.num_classes, img_size=opts.trainsize)


    opts.prediction_dir+=opts.model+'/'
    if not os.path.exists(opts.prediction_dir):
        os.makedirs(opts.prediction_dir, exist_ok=True)
    print('opts:',opts)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        try:
            model.load_state_dict(checkpoint["model_state"])
            model=model.cuda()
            print('try: load pth from:', opts.ckpt)
        except:
            dic = collections.OrderedDict()
            for k, v in checkpoint["model_state"].items():
                #print( k)
                mlen=len('module')+1
                newk=k[mlen:]
                # print(newk)
                dic[newk]=v
            model.load_state_dict(dic)
            model=model.cuda()
            print('except: load pth from:', opts.ckpt)
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)
   
    model.eval()
    for batch in tqdm(test_loader):
        imgs,  name=batch['img'], batch['name']
        imgs = imgs.type(torch.FloatTensor)

        imgs = Variable(imgs).cuda()
       
        preds = model(imgs)[0].max(1)[1].cpu()
   
        for i in range(preds.shape[0]):

            pred=preds[i]
            # print('pred.shape:', pred.shape)
            pred = normPRED(pred)
    
            save_output(name[i], pred, opts.prediction_dir)

       

if __name__ == '__main__':
    main()
