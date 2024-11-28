import time

import utils
from utils import *


def train_ps(args, model, early_stop, optimizer, logger, get_ps=True):
    epochs = 30
    train_loader = get_loader(args, stage='train')
    valid_loader = get_loader(args, stage='valid')
    # results
    train_loss = []
    valid_loss = []
    # training steps
    ps_results = {}
    for epoch in range(1, epochs + 1):
        model = model.train()
        # train epoch
        train_loss_epoch = []
        pbar = tqdm(enumerate(train_loader), disable=args.bar)
        pbar.set_description(f"Training Epoch {epoch}")
        for ind, data_batch in pbar:
            # optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None
            interaction, ngb_pos, ngb_neg = data_batch[0], data_batch[1], data_batch[2]
            # to device
            for l_key in ngb_pos:  # ngb_0 ngb_1 (k-hop ngb)
                for e_key in ngb_pos[l_key]:  # pos or neg
                    for i in range(len(ngb_pos[l_key][e_key])):  # 6 info of neighbor nodes
                        ngb_pos[l_key][e_key][i] = ngb_pos[l_key][e_key][i].to(args.device)
                        if args.n_channels == 2:
                            ngb_neg[l_key][e_key][i] = ngb_neg[l_key][e_key][i].to(args.device)
            for key in interaction:
                interaction[key] = interaction[key].to(args.device)

            if args.n_channels == 2:
                mse_loss, pred_ratings, ps_loss, ps, _ = model(interaction, ngb_pos, ngb_neg)
            else:
                mse_loss, pred_ratings, ps_loss, ps = model(interaction, ngb_pos, ngb_neg)
            """
                1. pre-train with biased mse and ps_loss 
                2. train with debias mse loss ad biased mse 
            """
            if get_ps:
                user_b = interaction['user'].cpu().numpy() - args.offset[0]
                item_b = interaction['item'].cpu().numpy() - args.offset[1]
                time_b = interaction['ts'].cpu().numpy()
                ps_in = ps.cpu().detach().numpy()
                for i in range(len(user_b)):
                    key = str(user_b[i]) + "_" + str(item_b[i]) + "_" + str(time_b[i])
                    # if key in ps_results:
                    #     print(key, epoch, ind, i)
                    #     print(ps_results[key])
                    #     print(ps_in[i])
                    #     print(interaction['ts'].cpu().numpy()[i])
                    #     sys.exit()
                    ps_results[key] = ps_in[i]

            pbar.set_postfix(mse_loss=mse_loss.mean().item(), ps_loss=ps_loss.item())
            train_loss_epoch.append(ps_loss.item())
            ps_loss.backward()
            optimizer.step()

        train_loss.append(np.mean(train_loss_epoch))

        # validation epoch
        val_loss = eval_classification(model, valid_loader, args.device, show_bar=args.bar,
                                       n_channels=args.n_channels)
        valid_loss.append(val_loss)
        # log info
        cur_lr = get_lr(optimizer)
        logger.info(f"Epoch {epoch}: ps loss:(x-entropy) {train_loss[-1]}, "
                    f"validation x-en: {val_loss}, lr: {cur_lr}")

        early_stop(val_loss, model, logger)
        if early_stop.early_stop:
            logger.info(f'Early stopping at {epoch} epoch...')
            break
    if get_ps:
        return ps_results


def train_round(args, model, early_stop, optimizer, logger, stage="pre1",
                is_ps_in=False, ps_input=None):
    if stage == "pre2":
        epochs = 10
    else:
        epochs = args.epochs
    # results
    train_loss = []
    valid_loss = []
    if ps_input is None:
        ps_input = {}

    if is_ps_in:
        valid_loader = get_loader(args, stage='valid')
        _, ps_dict = eval_one_epoch(model, valid_loader, args.device, show_bar=args.bar,
                                    test=False, num_val=args.num_val, stage=stage,
                                    n_channels=args.n_channels, is_ps_in=True, offset=args.offset)
        # fix ps_dict
        # ps_input.update(ps_dict)

        model.set_version(4)  # include ps input in the GNN aggregation step
        # train_loader = get_loader(args, stage='train', ps=ps_input)
        # valid_loader = get_loader(args, stage='valid', ps=ps_input)
    else:
        train_loader = get_loader(args, stage='train')
        valid_loader = get_loader(args, stage='valid')

    # training steps
    for epoch in range(1, epochs + 1):
        if is_ps_in:   # update ps_dict every xx epochs
            # print(list(ps_input.keys())[0])
            # print(ps_input['1_3406'])
            train_loader = get_loader(args, stage='train', ps=ps_input)
            valid_loader = get_loader(args, stage='valid', ps=ps_input)
            model.set_version(4)  # so that the ps is activated in local aggregation
        model = model.train()
        # train epoch
        train_loss_epoch = []
        pbar = tqdm(enumerate(train_loader), disable=args.bar)
        pbar.set_description(f"Training Epoch {epoch}")
        for ind, data_batch in pbar:
            # optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None
            interaction, ngb_pos, ngb_neg = data_batch[0], data_batch[1], data_batch[2]
            # to device
            for l_key in ngb_pos:  # ngb_0 ngb_1 (k-hop ngb)
                for e_key in ngb_pos[l_key]:  # pos or neg
                    for i in range(len(ngb_pos[l_key][e_key])):  # 6 info of neighbor nodes
                        ngb_pos[l_key][e_key][i] = ngb_pos[l_key][e_key][i].to(args.device)
                        if args.n_channels == 2:
                            ngb_neg[l_key][e_key][i] = ngb_neg[l_key][e_key][i].to(args.device)
            for key in interaction:
                interaction[key] = interaction[key].to(args.device)
            if args.n_channels == 2:
                mse_loss, pred_ratings, ps_loss, ps, cos_loss = model(interaction, ngb_pos, ngb_neg)
            else:
                mse_loss, pred_ratings, ps_loss, ps = model(interaction, ngb_pos, ngb_neg)
                cos_loss = ps_loss
            """
                1. pre-train with biased mse and ps_loss 
                2. train with debias mse loss ad biased mse 
            """
            # if is_ps_in:
            #     user_b = interaction['user'].cpu().numpy() - args.offset[0]
            #     item_b = interaction['item'].cpu().numpy() - args.offset[1]
            #     time_b = interaction['ts'].cpu().numpy()
            #     ps_in = ps.cpu().detach().numpy()
            #     for i in range(len(user_b)):
            #         key = str(user_b[i]) + "_" + str(item_b[i]) + "_" + str(time_b[i])
            #         ps_input[key] = ps_in[i]

            if stage == "pre1":  # train main module
                mse_loss = torch.mean(mse_loss)
                if args.n_channels == 2:  # disentangled embeds and cosine similarity constraints
                    if epoch < 4:
                        alpha = 0.01
                    else:
                        alpha = 0.005
                    final_loss = mse_loss    # + cos_loss * alpha
                    pbar.set_postfix(train_loss=final_loss.item(), mse_l=mse_loss.item(), ps_l=ps_loss.item(),
                                     cos_l=cos_loss.item())
                else:
                    final_loss = mse_loss
                    pbar.set_postfix(train_loss=final_loss.item(),
                                     mse_l=mse_loss.item(), ps_l=ps_loss.item())
                train_loss_epoch.append(mse_loss.item())
            elif stage == "pre2":  # train PS module
                final_loss = ps_loss
                pbar.set_postfix(mse_loss=mse_loss.mean().item(), ps_loss=ps_loss.item())
                train_loss_epoch.append(ps_loss.item())
            elif stage == "ps_l":   # train the with ps_loss (fix ps module or not?)
                mse_loss_ps = mse_loss / ps
                mse_loss_ps = torch.mean(mse_loss_ps)
                if args.n_channels == 2:
                    # final_loss = mse_loss_ps    # + cos_loss
                    pbar.set_postfix(mse_loss=mse_loss.mean().item(),
                                     mse_loss_ps=mse_loss_ps.item(),
                                     ps_loss=ps_loss.item(),
                                     cos_loss=cos_loss.item())
                else:
                    pbar.set_postfix(mse_loss=mse_loss.mean().item(),
                                     mse_loss_ps=mse_loss_ps.item(),
                                     ps_loss=ps_loss.item())
                # if epoch % 2 == 0:
                #
                #     final_loss = mse_loss_ps + 0.1 * ps_loss + mse_loss.mean() * 4
                # else:
                #     final_loss = mse_loss_ps
                final_loss = mse_loss_ps   # + 0.1 * ps_loss  # + mse_loss.mean() * 4
                train_loss_epoch.append(mse_loss.mean().item())
            elif stage == "naive":
                mse_loss_ps = mse_loss / interaction['rate_p']
                final_loss = torch.mean(mse_loss_ps)
                pbar.set_postfix(mse_loss=mse_loss.mean().item(),
                                 mse_loss_ps=final_loss.item())
            else:  # input sequence with ps score
                print("haven't implemented")
                sys.exit()
            # final_loss = loss
            final_loss.backward()
            optimizer.step()

        train_loss.append(np.mean(train_loss_epoch))

        # validation epoch
        if is_ps_in:
            val_loss, ps_dict = eval_one_epoch(model, valid_loader, args.device, show_bar=args.bar,
                                               test=False, num_val=args.num_val, stage=stage,
                                               n_channels=args.n_channels, is_ps_in=True, offset=args.offset)
            # ps_input.update(ps_dict)
        else:
            val_loss = eval_one_epoch(model, valid_loader, args.device, show_bar=args.bar,
                                      test=False, num_val=args.num_val, stage=stage,
                                      n_channels=args.n_channels)
        valid_loss.append(val_loss)
        # log info
        cur_lr = get_lr(optimizer)
        logger.info(f"Epoch {epoch}: train mse: {train_loss[-1]}, validation mse: {val_loss}, lr: {cur_lr}")
        early_stop(val_loss, model, logger)
        if early_stop.early_stop:
            logger.info(f'Early stopping at {epoch} epoch...')
            break
    return ps_input


def test_model(model, model_path, args, out_main, logger, stage="pre1", ps=None):
    # load the best model
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model = model.to(args.device)
    # test data
    if ps is not None:
        test_loader = get_loader(args, stage='test', ps=ps)
    else:
        test_loader = get_loader(args, stage='test')
    # run test
    test_file = f"{out_main}/test_org-{stage}-{time.time()}.csv"
    macro_res, weighted_res, mse_res, mae_res = eval_one_epoch(model, test_loader, args.device,
                                                               show_bar=args.bar, test=True,
                                                               out_file=test_file, num_val=args.num_val,
                                                               n_channels=args.n_channels)
    logger.info(f"test macro mse: {macro_res[0]}, macro mae: {macro_res[1]}")
    logger.info(f"test weighted mse: {weighted_res[0]}, weighted mae: {weighted_res[1]}")

    with open(test_file.replace(".csv", f".txt"), 'w') as fout:
        fout.write(f"test macro mse and mae: {macro_res[0]}\t{macro_res[1]}\n")
        fout.write(f"test weighted mse and mae : {weighted_res[0]}\t{weighted_res[1]}\n")
        fout.write(f"test org mse: {mse_res}\n")
        fout.write(f"test org mae: {mae_res}\n")

#
# def get_ps_all(model_t, adj):
#     model_t = model_t.eval()
#     for src_k in adj:
#         for tgt_k in adj[src_k]:
#             for i in range(len(adj[src_k][tgt_k])):
#

