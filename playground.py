from models.deeplab_resnet import *

if __name__ == '__main__':
    # block = Bottleneck
    backbone = 'ResNet34'
    # tasks_num_class = [40, 3]
    # tasks_num_class = [19, 1]
    # tasks_num_class = [40, 3, 1]
    tasks_num_class = [17, 3, 1, 1, 1]
    if backbone == 'ResNet18':
        layers = [2, 2, 2, 2]
        block = BasicBlock
    elif backbone == 'ResNet34':
        block = BasicBlock
        layers = [3, 4, 6, 3]
    elif backbone == 'ResNet101':
        block = Bottleneck
        layers = [3, 4, 23, 3]
    else:
        raise ValueError('backbone %s is invalid' % backbone)

    # block, layers, num_classes_tasks, init_method, init_neg_logits=None, skip_layer=0
    net = MTL2_Backbone(block, layers, tasks_num_class, 'equal')

    img = torch.ones((1, 3, 224, 224))

    # outs, policys = net(img, 5, True)
    count_params(net.backbone)

    import numpy as np
    if len(tasks_num_class) == 3:
        policy1 = np.array([1, 1, 1, 1,
                   1, 1, 0, 1,
                   1, 1, 1, 1,
                   1, 1, 1, 1,])
        policy1 = np.stack([policy1, 1 - policy1], axis=1).astype('float')
        policy1 = torch.from_numpy(policy1).cuda()
        policy2 = np.array([1, 1, 1, 1,
                            1, 1, 1, 1,
                            1, 0, 0, 0,
                            1, 1, 1, 1, ])
        policy2 = np.stack([policy2, 1 - policy2], axis=1).astype('float')
        policy2 = torch.from_numpy(policy2).cuda()

        policy3 = np.array([1, 1, 1, 1,
                            1, 0, 1, 1,
                            1, 1, 0, 0,
                            1, 1, 1, 1, ])
        policy3 = np.stack([policy3, 1 - policy3], axis=1).astype('float')
        policy3 = torch.from_numpy(policy3).cuda()
        policys = [policy1, policy2, policy3]

    elif len(tasks_num_class) == 2 and backbone == 'ResNet34':
        policy1 = np.array([1, 1, 1, 1,
                   1, 0, 0, 1,
                   0, 1, 1, 1,
                   0, 1, 1, 1,])
        policy1 = np.stack([policy1, 1 - policy1], axis=1).astype('float')
        policy1 = torch.from_numpy(policy1).cuda()
        policy2 = np.array([1, 1, 0, 1,
                            1, 1, 1, 1,
                            1, 0, 1, 1,
                            1, 1, 1, 1, ])
        policy2 = np.stack([policy2, 1 - policy2], axis=1).astype('float')
        policy2 = torch.from_numpy(policy2).cuda()
        policys = [policy1, policy2]
    elif len(tasks_num_class) == 2 and backbone == 'ResNet18':
        policy1 = np.array([1, 0, 1, 1,
                   1, 1, 1, 1])
        policy1 = np.stack([policy1, 1 - policy1], axis=1).astype('float')
        policy1 = torch.from_numpy(policy1).cuda()
        policy2 = np.array([1, 1, 1, 1,
                            1, 0, 1, 1 ])
        policy2 = np.stack([policy2, 1 - policy2], axis=1).astype('float')
        policy2 = torch.from_numpy(policy2).cuda()
        policys = [policy1, policy2]
    elif len(tasks_num_class) == 5:
        policy1 = np.array([1, 1, 1, 1,
                   1, 1, 1, 1,
                   1, 1, 1, 1,
                   1, 1, 1, 1,])
        policy1 = np.stack([policy1, 1 - policy1], axis=1).astype('float')
        policy1 = torch.from_numpy(policy1).cuda()
        policy2 = np.array([1, 1, 1, 1,
                            1, 0, 1, 1,
                            1, 1, 1, 1,
                            0, 1, 1, 1, ])
        policy2 = np.stack([policy2, 1 - policy2], axis=1).astype('float')
        policy2 = torch.from_numpy(policy2).cuda()
        policy3 = np.array([1, 1, 1, 1,
                   0, 1, 0, 1,
                   1, 1, 1, 1,
                   1, 1, 1, 1,])
        policy3 = np.stack([policy3, 1 - policy3], axis=1).astype('float')
        policy3 = torch.from_numpy(policy3).cuda()
        policy4 = np.array([1, 1, 1, 0,
                   1, 1, 1, 1,
                   1, 1, 1, 1,
                   1, 1, 1, 1,])
        policy4 = np.stack([policy4, 1 - policy4], axis=1).astype('float')
        policy4 = torch.from_numpy(policy4).cuda()
        policy5 = np.array([1, 1, 1, 1,
                   1, 1, 1, 1,
                   1, 0, 0, 1,
                   0, 1, 1, 1,])
        policy5 = np.stack([policy5, 1 - policy5], axis=1).astype('float')
        policy5 = torch.from_numpy(policy5).cuda()
        policys = [policy1, policy2, policy3, policy4, policy5]
    else:
        raise ValueError

    setattr(net, 'policys', policys)

    times = []
    input_dict = {'temperature': 5, 'is_policy': True, 'mode': 'fix_policy'}
    net.cuda()
    for _ in tqdm.tqdm(range(1000)):
        start_time = time.time()
        img = torch.rand((1, 3, 224, 224)).cuda()
        net(img, **input_dict)
        times.append(time.time() - start_time)

    print('Average time = ', np.mean(times))

    gflops = compute_flops(net, img.cuda(), {'temperature': 5, 'is_policy': True, 'mode': 'fix_policy'})
    print('Number of FLOPs = %.2f G' % (gflops / 1e9 / 2))
    pdb.set_trace()