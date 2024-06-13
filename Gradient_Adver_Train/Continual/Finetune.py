import torch
import torch.nn.functional as F
from utils.context import ctx_noparamgrad_and_eval
import pdb
def Finetune(train_loader,model,attacker,optimizer,args,scheduler,accs,accs_adv,losses):
    for i, data in enumerate(train_loader):
        # print('i: ',i)
        imgs, labels = data[0],data[1]
        # labels = label_transform(labels,experience.classes_in_this_experience)
        # print(labels)
        imgs, labels = imgs.cuda(), labels.cuda()
        # pdb.set_trace()
        # pdb.set_trace()
        # generate adversarial images:

        with ctx_noparamgrad_and_eval(model):
            imgs_adv = attacker.attack(model, imgs, labels)
        logits_adv = model(imgs_adv.detach())
        # logits for clean imgs:
        logits = model(imgs)

        # loss and update:
        loss = F.cross_entropy(logits, labels)

        if args.Lambda != 0:
            loss = (1-args.Lambda) * loss + args.Lambda * F.cross_entropy(logits_adv, labels)

        current_lr = scheduler.get_lr()[0]
        # metrics:
        accs.append((logits.argmax(1) == labels).float().mean().item())
        
        if args.Lambda != 0:
            accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())
        else:
            accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

    return train_loader, model, attacker, optimizer, args, scheduler, accs, accs_adv, losses