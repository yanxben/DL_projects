from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        # training parameters
        parser.add_argument('--epoch_start', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--epochs', type=int, default=2000, help='# of epochs to train')
        parser.add_argument('--epochs_pretrain', type=int, default=0, help='# of epochs to train')
        parser.add_argument('--pretrain_mode', type=str, default='default', help='# of epochs to train')
        parser.add_argument('--train_scheduler', type=str, default='mix,mix', help='list of string from train modes (default, mix, recon)')
        parser.add_argument('--default_train_mode', type=str, default='recon', help='default train mode (mix, recon)')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')

        self.isTrain = True
        return parser
