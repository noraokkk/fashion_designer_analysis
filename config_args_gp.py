import os

def get_args(parser,eval=False):
    parser.add_argument('--dataset', type=str, choices=['fashion_designers_gp'], default='fashion_designers_gp')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--results_dir', type=str, default='results/')
    parser.add_argument('--num_classes', type=int, default=5, help='class num')
    parser.add_argument('--freeze_backbone', action='store_true', default=True)

    # Optimization
    parser.add_argument('--optim', type=str, choices=['adam', 'sgd'], default='sgd')
    parser.add_argument('--lr', type=float, default=0.002)
    # parser.add_argument('--lr', type=float, default=0.002) #personality
    parser.add_argument('--batch_size', type=int, default=32) #personality
    parser.add_argument('--test_batch_size', type=int, default=-1)
    parser.add_argument('--grad_ac_steps', type=int, default=1)
    parser.add_argument('--scheduler_step', type=int, default=1000)
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--int_loss', type=float, default=0.0)
    parser.add_argument('--aux_loss', type=float, default=0.0)
    parser.add_argument('--loss_type', type=str, choices=['bce', 'mixed','class_ce','soft_margin'], default='bce')
    parser.add_argument('--scheduler_type', type=str, choices=['plateau', 'step'], default='plateau')
    parser.add_argument('--loss_labels', type=str, choices=['all', 'unk'], default='all')
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    # parser.add_argument('--weight_decay', type=float, default=2.551918610962277e-09)
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--max_batches', type=int, default=-1)
    parser.add_argument('--warmup_scheduler', action='store_true',help='')

    # # # Testing Models
    parser.add_argument('--inference', action='store_true', default=True) #True for evluation
    parser.add_argument('--saved_model_name', type=str, default='results//best_model.pt') #model3 resnet50
    parser.add_argument('--saved_backbone_name', type=str, default='results/fashion_designers_c5.3layer.bsz_128sz_224.sgd0.002/best_model.pt') #model3 resnet50

    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--name', type=str, default='')

    parser.add_argument('--image_size', type=int, default=224)
    # parser.add_argument('--data_dir', type=str, default='/home/sicelukwanda/modm/datasets/fashion_designers_list', help='the dir of the data json files')
    parser.add_argument('--data_dir', type=str, default='F:/datasets//fashion_designers_list', help='the dir of the data json files')
    args = parser.parse_args("")
    model_name = args.dataset


    model_name += '.bsz_{}sz_{}'.format(int(args.batch_size * args.grad_ac_steps),args.image_size) #personality img size
    model_name += '.'+args.optim+str(args.lr)#.split('.')[1]

    if args.name != '':
        model_name += '.'+args.name

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    model_name = os.path.join(args.results_dir,model_name)

    args.model_name = model_name


    if args.inference:
        args.epochs = 1


    if os.path.exists(args.model_name) and (not args.overwrite) and (not 'test' in args.name) and (not eval) and (not args.inference) and (not args.resume):
        print(args.model_name)
        # overwrite_status = input('Already Exists. Overwrite?: ')
        overwrite_status = 'y'
        overwrite_status = 'rm'
        if overwrite_status == 'rm':
            os.system('rm -rf '+args.model_name)
        elif not 'y' in overwrite_status:
            exit(0)
    elif not os.path.exists(args.model_name):
        os.makedirs(args.model_name)


    return args
