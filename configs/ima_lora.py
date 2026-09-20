
def get_args_parser(subparsers):
    subparsers.add_argument('--batch-size', default=24, type=int, help='Batch size per device')
    subparsers.add_argument('--epochs', default=5, type=int)

    # Model parameters
    subparsers.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', help='Name of model to train')
    subparsers.add_argument('--original_model', default='vit_base_patch16_224', type=str, metavar='ORIGINAL MODEL', help='Name of model to inference task id')
    subparsers.add_argument('--input-size', default=224, type=int, help='images input size')
    subparsers.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    subparsers.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    subparsers.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

    # Optimizer parameters
    subparsers.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    subparsers.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    subparsers.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    subparsers.add_argument('--clip-grad', type=float, default=2.0, metavar='NORM',  help='Clip gradient norm (default: None, no clipping)')
    subparsers.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    subparsers.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    subparsers.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')

    # Learning rate schedule parameters
    subparsers.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
    subparsers.add_argument('--lr', type=float, default=0.03, metavar='LR', help='learning rate (default: 0.03)')
    subparsers.add_argument('--init_lr', type=float, default=0.03, metavar='INITLR', help='init learning rate (default: 0.03)')
    subparsers.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    subparsers.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    subparsers.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    subparsers.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    subparsers.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    subparsers.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    subparsers.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    subparsers.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    subparsers.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    subparsers.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
    subparsers.add_argument('--unscale_lr', type=bool, default=True, help='scaling lr by batch size (default: True)')
    subparsers.add_argument('--milestones', default=[15], type=int)
    subparsers.add_argument('--weight_decay', default=5e-4, type=float)

    # Augmentation parameters
    subparsers.add_argument('--color-jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (default: 0.3)')
    subparsers.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    subparsers.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    subparsers.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    subparsers.add_argument('--reprob', type=float, default=0.0, metavar='PCT', help='Random erase prob (default: 0.25)')
    subparsers.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    subparsers.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    # Data parameters
    subparsers.add_argument('--data-path', default='/local_datasets/', type=str, help='dataset path')
    subparsers.add_argument('--dataset', default='Split-Imagenet-A', type=str, help='dataset name')
    subparsers.add_argument('--shuffle', default=False, help='shuffle the data order')
    subparsers.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    subparsers.add_argument('--device', default='cuda', help='device to use for training / testing')
    subparsers.add_argument('--seed', default=42, type=int)
    subparsers.add_argument('--eval', action='store_true', help='Perform evaluation only')
    subparsers.add_argument('--num_workers', default=4, type=int)
    subparsers.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    subparsers.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    subparsers.set_defaults(pin_mem=True)

    # distributed training parameters
    subparsers.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    subparsers.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Continual learning parameters
    subparsers.add_argument('--num_tasks', default=10, type=int, help='number of sequential tasks')
    subparsers.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    subparsers.add_argument('--task_inc', default=False, type=bool, help='if doing task incremental')

    # lora parameters
    subparsers.add_argument('--lora_type', default='continual', choices=['continual', 'hide'],type=str, help='lora type')
    subparsers.add_argument('--lora_pool', action="store_true")
    subparsers.add_argument('--size', default=10, type=int,)
    subparsers.add_argument('--lora_rank', default=4, type=int,)
    subparsers.add_argument('--lora_momentum', default=0.1, type=float,)

    subparsers.add_argument('--con', default=0.1, type=float,)
    subparsers.add_argument('--K', default=5, type=int,)
    subparsers.add_argument('--En', default='gen', choices=['msp', 'gen'])
    subparsers.add_argument('--tau', default=-10, type=float)
    subparsers.add_argument('--task_routing_mode', default='class', choices=['class', 'task_energy'], type=str, help='route evaluation samples by top class or aggregated task energy')
    subparsers.add_argument('--task_routing_temperature', default=0.1, type=float)

    # ViT parameters
    subparsers.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
    subparsers.add_argument('--head_type', default='token', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
    subparsers.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')

    subparsers.add_argument('--crct_epochs', default=30, type=int)
    subparsers.add_argument('--train_inference_task_only', action='store_true')
    subparsers.add_argument('--original_model_mlp_structure', default=[2], type=int)
    subparsers.add_argument('--ca_lr', default=0.005, type=float)
    subparsers.add_argument('--trained_original_model', default='', type=str)
    subparsers.add_argument('--not_train_ca', action='store_true')
    subparsers.add_argument('--ca_epochs', default=30, type=int)
    subparsers.add_argument('--ca_storage_efficient_method', default='multi-centroid', choices=['covariance', 'multi-centroid', 'variance'], type=str)
    subparsers.add_argument('--n_centroids', default=2, type=int)
    subparsers.add_argument('--print_freq', type=int, default=10, help = 'The frequency of printing')

    # --- dau phep chieu ngau nhien va hop nhat hai tang ---
    subparsers.add_argument('--rp_activation', default='relu', choices=['relu', 'square', 'gelu', 'none'], type=str)
    subparsers.add_argument('--rp_bare_extractor', action='store_true', help="pin a freshly created PRETRAINED backbone as the RP extractor instead of the TII snapshot; original_model is not the raw ViT (its weights come from the TII checkpoint), so this is the only way to measure the no-first-session-adaptation condition")
    subparsers.add_argument('--rp_bare_model', default='', type=str, help="checkpoint name for the --rp_bare_extractor snapshot only; empty means use --original_model. Set it to vit_base_patch16_224_augreg_in1k to test whether the published no-first-session ImageNet-R numbers were measured on an ImageNet-1k fine-tuned backbone -- ImageNet-R is 200 ImageNet-1k classes redrawn, so that choice would lift ImageNet-R and leave CUB alone")
    subparsers.add_argument('--rp_calibrate', action='store_true', help='fit one scalar temperature on training scores so the head Loss is comparable to a softmax classifier (only the scalar is kept)')
    subparsers.add_argument('--rp_class_fusion_gate', default='none', type=str, choices=['none', 'margin', 'margin_both', 'entropy'], help="scale the class-fusion weight per sample by how undecided the routed head is; 'margin' uses its normalised top-2 margin, 'margin_both' additionally scales by the RP head's own margin so the blend opens only when the routed head is undecided AND the RP head is decided, 'entropy' uses normalised entropy, 'none' fuses every sample equally")
    subparsers.add_argument('--rp_class_fusion_min_tasks', default=1, type=int, help='only blend the RP head once this many tasks have been seen; its Gram estimate is unreliable with few classes and the blend costs accuracy in the first stages')
    subparsers.add_argument('--rp_class_fusion_sharpen', default=1.0, type=float, help='scale the fused class scores before the loss; blending flattens the winner margin and cross-entropy penalises that, and scaling is monotone so accuracy is unchanged')
    subparsers.add_argument('--rp_class_fusion_weight', default=0.0, type=float, help='weight on the RP head as a second classifier, blended with the routed HRM-PET head after DRM/CRM (0 = routed head only); both are standardized over valid classes first')
    subparsers.add_argument('--rp_dim', default=5000, type=int, help='random projection width M (RanPAC default 10000; 5000 halves the Gram cost)')
    subparsers.add_argument('--rp_feature_source', default='original', choices=['original', 'lora'], type=str, help='original = frozen TII backbone features (no routing at all); lora = one fixed HRM-PET adapter used as first-session adaptation, still routing-free')
    subparsers.add_argument('--rp_fusion_ramp', default=0.0, type=float, help='ramp both fusions with the number of tasks seen, as (seen/total)**ramp; 0 keeps the weight constant, and the ramp reaches 1.0 at the final stage so the reported final row is unchanged either way')
    subparsers.add_argument('--rp_fusion_ramp_scope', default='both', type=str, choices=['both', 'class', 'route'], help='which fusion the ramp applies to; the RP head is a weak classifier early but a strong router from the first stage, so ramping the route weight costs routing accuracy for nothing')
    subparsers.add_argument('--rp_head', action='store_true', help='routing-free RanPAC-style classifier: frozen random projection + nonlinearity, then ridge on the accumulated Gram matrix and class prototypes; bypasses the task-inference bottleneck entirely')
    subparsers.add_argument('--rp_input_norm', default='none', choices=['none', 'half', 'cifar_half'], type=str, help="input range for the RP head's frozen extractor: 'half' maps ToTensor()'s [0,1] to the [-1,1] the ImageNet-21K ViT checkpoint was trained on; the LoRA path is left untouched")
    subparsers.add_argument('--rp_lambda', default=10000.0, type=float, help='ridge regularizer added to the Gram diagonal before inversion')
    subparsers.add_argument('--rp_lambda_criterion', default='mse', type=str, choices=['mse', 'cosine', 'accuracy'], help="what the lambda sweep optimises; 'mse' is RanPAC's held-out squared error, 'cosine' is the scale-invariant Frobenius cosine between predictions and targets, which does not reward the magnitude shrinkage that large lambda buys and that the per-sample standardisation in the blend discards anyway; 'accuracy' scores held-out top-1 directly, which needs the current task's held-out features in memory but measures the quantity that is actually wanted -- on ImageNet-R the head's own accuracy peaks a decade below where squared error puts it")
    subparsers.add_argument('--rp_lambda_scope', default='task', type=str, choices=['task', 'pooled'], help="which data the lambda sweep ranges over; 'task' rebuilds the 80:20 split from scratch every task, which is RanPAC's own procedure, while 'pooled' lets it accumulate across every task seen so far. Pooling biases the choice upward: the norm of the Gram matrix grows linearly with sample count and the best ridge grows with that norm, so ten tasks pooled favour a larger lambda than one task alone. 'pooled' reproduces the earlier ImageNet-R measurement that selected 1e5 at every task")
    subparsers.add_argument('--rp_lambda_search', action='store_true', help="choose lambda per task the way RanPAC does, by splitting the current task 80:20 and sweeping 17 orders of magnitude against held-out squared error, instead of using the fixed --rp_lambda; old tasks are never touched and no features are stored, since the validation error is closed-form in the second-order statistics already kept")
    subparsers.add_argument('--rp_logit_blend', default=0.0, type=float, help='weight on the HRM-PET classifier log-probabilities blended into the head scores; free because it comes from the same adapter forward (0 = pure head)')
    subparsers.add_argument('--rp_lora_task', default=0, type=int, help='which task adapter provides the fixed feature space when rp_feature_source=lora (0 = first session, RanPAC-style); it must not vary per task or the Gram matrix mixes incompatible spaces')
    subparsers.add_argument('--rp_normalize', default='none', choices=['none', 'l2', 'scale'], type=str, help='feature scaling before the random projection; raw ViT pre_logits scale distorts the ReLU projection and the single ridge lambda')
    subparsers.add_argument('--rp_pin_extractor', action='store_true', help='freeze the task-1 TII snapshot as the RP feature extractor; without it the per-task checkpoint reload mixes incompatible feature spaces in the Gram matrix')
    subparsers.add_argument('--rp_route_audit', action='store_true', help='measure whether RP-head routing and TII routing fail on the same samples; the union bounds what fusing the two routers could recover')
    subparsers.add_argument('--rp_route_fusion', action='store_true', help='route by fusing RP-head and TII task scores, then classify with the routed task LoRA; the two routers fail on different samples so their union exceeds HRM-PET routing on every dataset measured')
    subparsers.add_argument('--rp_route_fusion_drm', action='store_true', help='hand the fused route to the existing DRM/CRM refinement instead of replacing it; keeps the stage that carries the baseline and improves only the routing that precedes it')
    subparsers.add_argument('--rp_route_fusion_ls_weight', default=0.0, type=float, choices=[0.0], help='legacy zero-valued argument retained for result-reproduction commands')
    subparsers.add_argument('--rp_route_fusion_weight', default=0.5, type=float, help='weight on TII in the fused task log-probabilities (0 = RP head only, 1 = TII only)')
    subparsers.add_argument('--rp_seed', default=1993, type=int, help='seed for the frozen random projection (regenerated, never stored)')
    subparsers.add_argument('--strict_exemplar_free', action='store_true', help='reject any configuration that stores per-example real features or rereads historical train images')
    subparsers.add_argument('--report_conventional_cost', action='store_true', help='log per-sample LoRA and forward-call cost of the default HRM-PET (DRM+CRM) evaluation path')
    subparsers.add_argument('--classifier_union_audit', action='store_true', help='measure whether the RP head, treated as a classifier, is correct where the routed HRM-PET head is wrong; bounds what a classifier-level fusion could add on top of the routing gain')

    # --- thuoc tinh ma trainers/engines doc thang tu args ---
    subparsers.add_argument('--max_train_tasks', default=0, type=int, help='stop after this many tasks while preserving the num_tasks dataset partition; 0 runs all tasks')
