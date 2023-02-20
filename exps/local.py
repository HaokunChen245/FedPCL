
def Local(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list):
    idxs_users = np.arange(args.num_users)
    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round} |\n')
        for idx in idxs_users:
            local_node = LocalUpdate(args=args, dataset=train_dataset_list[idx],idxs=user_groups[idx])
            w, loss = local_node.update_weights(idx, backbone_list=backbone_list, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            summary_writer.add_scalar('Train/Loss/user' + str(idx), loss, round)

        # update global weights
        local_weights_list = copy.deepcopy(local_weights)

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        if round % 10 == 0:
            with torch.no_grad():
                for i in range(args.num_users):
                    print('Test on user {:d}'.format(i))
                    local_test = LocalTest(args=args, dataset=test_dataset_list[i], idxs=user_groups_test[i])
                    local_model_list[i].eval()
                    acc, loss = local_test.test_inference(i, args, backbone_list, local_model_list[i])
                    summary_writer.add_scalar('Test/Acc/user' + str(i), acc, round)

    acc_mtx = torch.zeros([args.num_users])
    loss_mtx = torch.zeros([args.num_users])
    with torch.no_grad():
        for idx in range(args.num_users):
            print('Test on user {:d}'.format(idx))
            local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
            local_model_list[idx].eval()
            acc, loss = local_test.test_inference(idx, args, backbone_list, local_model_list[idx])
            acc_mtx[idx] = acc
            loss_mtx[idx] = loss

    print('For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(torch.mean(acc_mtx), torch.std(acc_mtx)))

    return acc_mtx