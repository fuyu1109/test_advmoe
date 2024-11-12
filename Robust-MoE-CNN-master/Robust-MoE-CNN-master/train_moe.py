from __future__ import absolute_import
from __future__ import print_function

import logging
import os
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from args import get_args_parser
from models.layers.router import build_router
from utils.eval_utils import std_val, adv_val, adv_val_router
from utils.general_utils import (
    save_checkpoint,
    set_router,
    parse_configs_file,
    create_save_dir, initialize_weights,
    set_seed, get_data_model, AverageMeter, split_data_and_move_to_device)
from utils.schedules import get_lr_policy, get_optimizer


def trainer(model, router, device, train_loader, epoch, optimizer, router_optimizer, args):#TRADES
    print(f" ->->->->->->->->->-> Epoch {epoch} with Adversarial training (TRADES) <-<-<-<-<-<-<-<-<-<-")

    losses = AverageMeter("Loss", ":.4f")
    losses_natural = AverageMeter("Loss-natural", ":.3f")
    losses_robust = AverageMeter("Loss-robust", ":.3f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    criterion = torch.nn.CrossEntropyLoss()

    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch} Training", ncols=120)
    model.train()
    router.train()
    for data in pbar:
        images, target = split_data_and_move_to_device(data, device)
        scores = model.router(images)#干净样本的路由器输出
        result = model(images)

        # define KL-loss
        criterion_kl = torch.nn.KLDivLoss(reduction="sum")
        model.eval()
        router.eval()
        batch_size = len(images)
        out_nat = model(images)

        x_adv = (images.detach() + 0.001 * torch.randn(images.shape).to(device).detach())
        #制作对抗样本
        for _ in range(args.num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                adv_scores = router(x_adv)
                out_adv = model(x_adv)
                ## 计算模型和路由器的KL损失
                loss_kl = criterion_kl(
                    F.log_softmax(out_adv, dim=1),
                    F.softmax(out_nat, dim=1),
                )
                loss_kl_router = criterion_kl(
                    F.log_softmax(adv_scores, dim=1),
                    F.softmax(scores, dim=1),
                )
                loss_kl += loss_kl_router

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, images - args.epsilon), images + args.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)

        model.train()
        router.train()

        x_adv = Variable(torch.clamp(x_adv, 0, 1))
        optimizer.zero_grad()

        logits_nat = model(images)
        loss_natural = criterion(logits_nat, target)#计算干净样本的损失

        logits_adv = model(x_adv)

        loss_robust = (1.0 / batch_size) * criterion_kl(
            F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1)
        )#计算对抗样本的KL损失

        loss = loss_natural + args.beta * loss_robust

        # measure get_accuracy and record loss
        with torch.no_grad():
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            losses_natural.update(loss_natural.item(), batch_size)
            losses_robust.update(loss_robust.item(), batch_size)
            top1.update(torch.argmax(result, 1).eq(target).float().mean().item(), batch_size)

        pbar.set_postfix_str(
            f"Source Acc {100 * top1.avg:.2f}%, Loss {losses_natural.avg:.5f}, Robust Loss{losses_robust.avg:.5f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # define KL-loss

        # 第二阶段
        criterion_kl = torch.nn.KLDivLoss(reduction="sum")
        model.eval()
        router.eval()
        batch_size = len(images)
        out_nat = model(images)

        x_adv = (images.detach() + 0.001 * torch.randn(images.shape).to(device).detach())
        for _ in range(args.num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                adv_scores = router(x_adv)
                out_adv = model(x_adv)

                loss_kl = criterion_kl(
                    F.log_softmax(out_adv, dim=1),
                    F.softmax(out_nat, dim=1),
                )
                loss_kl_router = criterion_kl(
                    F.log_softmax(adv_scores, dim=1),
                    F.softmax(scores, dim=1),
                )
                loss_kl += loss_kl_router

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, images - args.epsilon), images + args.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)

        model.train()
        router.train()

        x_adv = Variable(torch.clamp(x_adv, 0, 1))
        optimizer.zero_grad()

        logits_nat = model(images)
        loss_natural = criterion(logits_nat, target)

        adv_scores = router(x_adv)#获取对抗样本的路由器输出

        loss = args.alpha * criterion(scores, target % args.n_expert) + args.beta * (1.0 / batch_size) * criterion_kl(
            F.log_softmax(adv_scores, dim=1), F.softmax(scores, dim=1)
        )#在计算损失的时候结合了路由器的输出

        # measure get_accuracy and record loss
        with torch.no_grad():
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            losses_natural.update(loss_natural.item(), batch_size)
            losses_robust.update(loss_robust.item(), batch_size)
            top1.update(torch.argmax(result, 1).eq(target).float().mean().item(), batch_size)

        router_optimizer.zero_grad()
        loss.backward()
        router_optimizer.step()

        pbar.set_postfix_str(
            f"Source Acc {100 * top1.avg:.2f}%, Loss {losses_natural.avg:.5f}, Robust Loss{losses_robust.avg:.5f}")


def mart_trainer_v2(model, router, device, train_loader, epoch, optimizer, router_optimizer, args):
    print(f" ->->->->->->->->->-> Epoch {epoch} with Adversarial training (MART) <-<-<-<-<-<-<-<-<-<-")

    # 初始化损失和准确率记录器
    losses = AverageMeter("Loss", ":.4f")
    losses_bce = AverageMeter("Loss-BCE", ":.3f")
    losses_weighted_kl = AverageMeter("Loss-Weighted-KL", ":.3f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    criterion = torch.nn.CrossEntropyLoss()

    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch} Training", ncols=120)

    for data in pbar:
        images, labels = split_data_and_move_to_device(data, device)
        scores = model.router(images)  # 干净样本的路由器输出
        criterion_kl = torch.nn.KLDivLoss(reduction="sum")
        # 生成对抗样本
        model.eval()
        router.eval()

        logits_clean = model(images)
        out_nat = model(images)
        x_adv = (images.detach() + 0.001 * torch.randn(images.shape).to(device).detach())
        for _ in range(args.num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                adv_scores = router(x_adv)
                out_adv = model(x_adv)

                loss_kl = criterion_kl(
                    F.log_softmax(out_adv, dim=1),
                    F.softmax(out_nat, dim=1),
                )
                loss_kl_router = criterion_kl(
                    F.log_softmax(adv_scores, dim=1),
                    F.softmax(scores, dim=1),
                )
                loss_kl += loss_kl_router

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, images - args.epsilon), images + args.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)

        # 第一阶段：模型参数更新
        model.train()
        router.train()
        x_adv = Variable(torch.clamp(x_adv, 0, 1))
        optimizer.zero_grad()

        logits_adv = model(x_adv)
        probs_adv = F.softmax(logits_adv, dim=1)

        tmp1 = torch.argsort(probs_adv, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == labels, tmp1[:, -2], tmp1[:, -1])

        loss_bce_adv = F.cross_entropy(logits_adv, labels, reduction="none") + \
                       F.nll_loss(torch.log(1.0001 - probs_adv + 1e-12), new_y, reduction="none")

        #print(f"Loss (BCE, model): {loss_bce_adv.mean().item()}")

        probs_clean = F.softmax(logits_clean, dim=1)
        log_prob_adv = torch.log(probs_adv + 1e-12)
        loss_kl = torch.sum(
            F.kl_div(log_prob_adv, probs_clean, reduction="none"), dim=1
        )
        true_probs = torch.gather(probs_clean, 1, labels.unsqueeze(1)).squeeze()
        loss_weighted_kl = loss_kl * (1.0000001 - true_probs)

        loss = loss_bce_adv.mean() + args.beta * loss_weighted_kl.mean()#计算损失

        #print(f"Loss (combined, model): {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 第二阶段：路由器参数更新
        criterion_kl = torch.nn.KLDivLoss(reduction="sum")
        model.eval()
        router.eval()
        batch_size = len(images)
        out_nat = model(images)

        x_adv = (images.detach() + 0.001 * torch.randn(images.shape).to(device).detach())
        for _ in range(args.num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                adv_scores = router(x_adv)
                out_adv = model(x_adv)

                loss_kl = criterion_kl(
                    F.log_softmax(out_adv, dim=1),
                    F.softmax(out_nat, dim=1),
                )
                loss_kl_router = criterion_kl(
                    F.log_softmax(adv_scores, dim=1),
                    F.softmax(scores, dim=1),
                )
                loss_kl += loss_kl_router

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, images - args.epsilon), images + args.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)

        model.train()
        router.train()

        x_adv = Variable(torch.clamp(x_adv, 0, 1))
        optimizer.zero_grad()

        logits_nat = model(images)
        loss_natural = criterion(logits_nat, labels)

        adv_scores = router(x_adv)  # 获取对抗样本的路由器输出

        loss = args.alpha * criterion(scores, labels % args.n_expert) + args.beta * (1.0 / batch_size) * criterion_kl(
            F.log_softmax(adv_scores, dim=1), F.softmax(scores, dim=1)
        )  # 在计算损失的时候结合了路由器的输出

        # 路由器优化
        router_optimizer.zero_grad()
        loss.backward()
        router_optimizer.step()

        # 更新进度条显示
        with torch.no_grad():
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            losses_bce.update(loss_bce_adv.mean().item(), batch_size)
            losses_weighted_kl.update(loss_weighted_kl.mean().item(), batch_size)
            top1.update(torch.argmax(logits_clean, 1).eq(labels).float().mean().item(), batch_size)

        pbar.set_postfix_str(
            f"Source Acc {100 * top1.avg:.2f}%, Loss-BCE {losses_bce.avg:.5f}, Loss-Weighted-KL {losses_weighted_kl.avg:.5f}"
        )


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    if args.configs is not None:
        parse_configs_file(args)

    # create result dir (for logs, checkpoints, etc.)
    if args.evaluate:
        result_sub_dir = os.path.join("results", "evaluate")
    else:
        result_sub_dir = os.path.join("results", "training")
    result_sub_dir = os.path.join(result_sub_dir, os.path.basename(__file__.split('.')[0]))
    result_sub_dir = create_save_dir(args, result_sub_dir, special_prefix=args.exp_identifier)

    set_seed(args.seed)

    # Set logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a"))
    logger.info(args)

    # Select device
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare data and model
    model, train_loader, train_router_loader, test_loader, image_dim = get_data_model(args, device)
    initialize_weights(model)
    optimizer = get_optimizer(model, args)
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args.lr, args.epochs)

    router = build_router(num_experts=args.n_expert).to(device)
    set_router(model, router)
    router_optimizer = get_optimizer(router, args)
    router_lr_policy = get_lr_policy(args.lr_schedule)(router_optimizer, args.lr, args.epochs)

    # Record the best get_accuracy
    start_epoch = 0
    best_acc = 0
    sa_record = 0

    # resume (if checkpoint provided).
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            router.load_state_dict(checkpoint["router"])
            start_epoch = checkpoint["epoch"]
            best_acc = checkpoint["best_acc"]
            sa_record = checkpoint["sa_record"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            router_optimizer.load_state_dict(checkpoint["router_optimizer"])
            logger.info("=> resuming from '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
        else:
            raise ValueError("=> No checkpoint found at '{}' for resume, please double check!".format(args.resume))

    # if args.evaluate:
    #     sa = std_val(model, router, device, test_loader)
    #     ra = adv_val(model, router, device, test_loader, args)
    #     print(f"Evaluation Results: SA: {sa: .2f}%, RA: {ra: .2f}%.")
    #     ra_model, ra_router = adv_val_router(model, device, test_loader, args)
    #     print(f"Attacking Router: Evaluation Results: RA Router: {ra_router: .2f}%, RA Model: {ra_model: .2f}%.")
    #     return

    # Start training
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        lr_policy(epoch)
        router_lr_policy(epoch)

        # train
        mart_trainer_v2(model, router, device, train_loader, epoch, optimizer, router_optimizer, args)
        #trainer(model, router, device, train_loader, epoch, optimizer, router_optimizer, args)

        # sa = std_val(model, router, device, test_loader)
        # ra = adv_val(model, router, device, test_loader, args)
        # is_best = ra > best_acc
        # if is_best:
        #     best_acc = ra
        #     sa_record = sa
        # logger.info(
        #     f"Epoch {epoch}, SA: {sa: .2f}%, RA: {ra: .2f}%. [best performance (RA): {best_acc: .2f}, (SA): {sa_record: .2f}]"
        # )
        result_sub_dir = 'result/MART-ciafr10-lr0.008-noeval-epoch60'
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "router": router.state_dict(),
                "best_acc": best_acc,
                "sa_record": sa_record,
                "optimizer": optimizer.state_dict(),
                "router_optimizer": router_optimizer.state_dict()
            },
            #is_best,
            result_dir=os.path.join(result_sub_dir, "checkpoint"),
        )

        epoch_end_time = time.time()
        logger.info(f"Time consumption for current epoch is {(epoch_end_time - epoch_start_time):.2f}s")


if __name__ == "__main__":
    main()
