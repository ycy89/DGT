import torch
import argparse
from dataset_process import get_data
from model import SDHID

def init_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--neg_valid_num', default=199, type=int)
    parser.add_argument('--k', default=4, type=int)
    parser.add_argument('--dim', default=64, type=int)
    parser.add_argument('--dataset', default='cd', type=str)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--train_batch_size', default=512, type=int)
    parser.add_argument('--test_batch_size', default=512, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--valid_interval', default=5, type=int)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--iter_time', default=5, type=int)
    parser.add_argument('--head_num', default=8, type=int)
    parser.add_argument('--block', default=1, type=int)
    parser.add_argument('--reg_weight', type=float, default=1e-5)
    parser.add_argument('--CL_weight', type=float, default=1e-4)
    parser.add_argument('--dCov_weight', type=float, default=1e-4)

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = init_setting()
    train_loader, test_loader, uvs, ivs, train_graph = get_data(args)
    model = SDHID(uvs, ivs, train_graph, args).to(args.device)
    opt = torch.optim.Adam(model.parameters(), args.lr)

    best_result = [1e3, 1e3, 1e3, 1e3]
    for epoch in range(args.epochs):
        model.train()
        avg_loss = []
        for user, seq, pos, rate in train_loader:
            user, seq, pos, rate = user.to(args.device), seq.to(args.device), pos.to(args.device), rate.to(args.device)
            loss = model.calc_loss(user, seq, pos, rate)
            opt.zero_grad()
            loss.backward()
            opt.step()
            avg_loss.append(loss.item())
        print(f'Epoch {epoch+1}  train_loss:{sum(avg_loss) / len(avg_loss)}')

        if (epoch + 1) % args.valid_interval == 0:
            model.eval()
            st = 0
            bs = 1024
            user_embs = torch.zeros([uvs, args.dim]).to(args.device)
            # item_embs = torch.zeros([ivs + 1, args.dim]).to(args.device)
            all_logits = torch.zeros(uvs).to(args.device)
            all_ratings = torch.zeros(uvs).to(args.device)
            item_embs = model.predict_emb()

            for user, seq, target, rate in test_loader:
                user, seq, target, rate = user.to(args.device), seq.to(args.device), target.to(args.device), rate.to(args.device)
                seq_emb = item_embs[seq]
                target_emb = item_embs[target]
                timeline_mask = (seq == model.mask_value).to(torch.bool)
                all_logits[user] = model.predict_logits(seq_emb, target_emb, timeline_mask)
                all_ratings[user] = rate

            macro_mse, macro_mae, weighted_mse, weighted_mae = model.rating_pred_loss_level(all_logits, all_ratings)
            if best_result[0] > macro_mse and best_result[2] > weighted_mse:
                best_result = [macro_mse, macro_mae, weighted_mse, weighted_mae]
            print(f'macro_mse:{macro_mse}, macro_mae:{macro_mae}')
            print(f'weighted_mse:{weighted_mse}, weighted_mae:{weighted_mae}')

    print(f'best_result:')
    print(f'macro_mse:{best_result[0]}, macro_mae:{best_result[1]}')
    print(f'weighted_mse:{best_result[2]}, weighted_mae:{best_result[3]}')





