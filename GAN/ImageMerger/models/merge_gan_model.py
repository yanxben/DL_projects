import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import encoder_decoder

#class CycleGANModel(BaseModel):
class mergeganmodel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--sep', type=int, default=128, help='')
            parser.add_argument('--input_size', type=int, default=96, help='')
            parser.add_argument('--last_conv_nc', type=int, default=512, help='')
            parser.add_argument('--lambda_G1', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_G2', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_Background', type=float, default=0.5, help='use background mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--background', action='store_false',
                                help='use background')
            parser.add_argument('--lambda_Identity', type=float, default=1.0,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--attention', action='store_true',
                                help='use attention layer')

            parser.add_argument('--preprocess_mask', action='store_false',
                                help='use preprocessing on mask')
            parser.add_argument('--preprocess_flip', action='store_false',
                                help='use preprocessing horizontal flip')
            parser.add_argument('--preprocess_shift', action='store_false',
                                help='use preprocessing shifting')
            parser.add_argument('--preprocess_rotate', action='store_false',
                                help='use preprocessing rotation')
            parser.add_argument('--preprocess_permute', action='store_false',
                                help='use preprocessing color permutation')
            parser.add_argument('--preprocess_jitter', action='store_false',
                                help='use preprocessing color jitter')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'rec_A', 'background_A', 'D_B', 'G_B', 'rec_B', 'background_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_Background > 0.0:  # if identity loss is used, we also visualize background_loss A[1-mask]=G(A,B)[1-mask] and B[1-mask]=G(B,A)[1-mask]
            visual_names_A.append('background_A')
            visual_names_B.append('background_B')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Generator
            self.model_names = ['G']

        # define networks (both Generators and discriminators)
        self.A = 0
        self.B = 1

        #self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netG = encoder_decoder.Generator(opt.input_nc, opt.input_nc, opt.last_conv_nc, opt.sep, opt.input_size, opt.background, False).to(self.device)
        if self.isTrain:  # define discriminator
            #self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
            #                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD = encoder_decoder.Discriminator(opt.input_nc, opt.last_conv_nc, opt.input_size).to(self.device)

        if self.isTrain:
            if False:  #self.opt.use_pool and False:
                self.fake_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            #self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionBackground = torch.nn.L1Loss()  #networks.BackgroundLoss()
            self.criterionCycle = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, model_input, input_mode):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        # We separate the train domain to 2 groups.
        # Real_G are training images for the generator.
        # Real_D are reference images for the descriminator.
        # This is crucial so that the descriminator doesn't learn to focus on learning the real image and detect
        # that it was altered.

        self.real_G = model_input['real_G'].clone()
        self.real_D = model_input['real_D'].clone()
        N1, C1, H1, W1 = self.real_G.shape
        N2, C2, H2, W2 = self.real_D.shape

        if self.opt.background:
            if not self.opt.attention:
                self.mask_G = model_input['mask_G'].clone()
                self.mask_D = model_input['mask_D'].clone()
                self.mask_G = self.mask_D.unsqueeze(1).expand_as(self.real_G)
                self.mask_D = self.mask_D.unsqueeze(1).expand_as(self.real_D)

        self.input_mode = input_mode
        if self.input_mode == 'mix':
            self.mask_G = self.mask_G.reshape(-1, 2, C1, H1, W1)
            self.real_G = self.real_G.reshape(-1, 2, C1, H1, W1)
        elif self.input_mode == 'reflection':
            self.mask_G = self.mask_G.unsqueeze(1).expand(N1, 2, C1, H1, W1)
            self.real_G = self.real_G.unsqueeze(1).expand(N1, 2, C1, H1, W1)

        if self.opt.preprocess_mask:
            self.preprocess()
            self.real_G_ref = ((self.real_G_ref - 0.5) * 2).to(self.device)

        # To device
        self.real_G = ((self.real_G - 0.5) * 2).to(self.device)
        self.real_D = ((self.real_D - 0.5) * 2).to(self.device)
        if self.opt.background:
            if not self.opt.attention:
                self.mask_G = self.mask_G.to(self.device)
                self.mask_D = self.mask_D.to(self.device)

        #self.image_paths_G = model_input['paths_G']
        #self.image_paths_D = model_input['paths_D']

    def preprocess(self, opt=None):
        # This function preprocesses the inputs with augmentation.
        # 1. Horizontal flip
        # 2. Shift
        # 3. Rotate
        # 4. Permute colors
        # 5. brightness, Contrast

        if opt is None:
            opt = self.opt

        N1, B1, C1, H1, W1 = self.real_G.shape
        N2, C2, H2, W2 = self.real_D.shape
        # 1. Flip
        if opt.preprocess_flip:
            flip = torch.randint(2, (N1, B1, 1, 1, 1)).expand_as(self.real_G)
            self.real_G = torch.where(flip == 1, self.real_G.flip(4), self.real_G)
            self.mask_G = torch.where(flip == 1, self.mask_G.flip(4), self.mask_G)
            flip = torch.randint(2, (N2,1, 1, 1)).expand_as(self.real_D)
            self.real_D = torch.where(flip == 1, self.real_D.flip(3), self.real_D)
            self.mask_D = torch.where(flip == 1, self.mask_D.flip(3), self.mask_D)

        # 2. Shift
        # 3. Rotate

        self.mask_G = torch.where(self.mask_G >= 0.5, torch.ones_like(self.mask_G), torch.zeros_like(self.mask_G))
        self.mask_D = torch.where(self.mask_D >= 0.5, torch.ones_like(self.mask_D), torch.zeros_like(self.mask_D))

        self.real_G_ref = self.real_G.clone()
        #self.real_D_ref = self.real_D.clone()

        # 4. Permute colors
        if opt.preprocess_permute:
            permute1 = torch.randint(10, (N1, B1)) >= 5
            permute2 = torch.randint(10, (N1, B1)) >= 5
            permute3 = torch.randint(10, (N1, B1)) >= 5
            # self.real_G[permute1][self.mask_G[permute1] == 1] = self.real_G[permute1][self.mask_G[permute1] == 1][:, :, (0, 2, 1), :, :]
            #self.real_G[permute2][self.mask_G[permute2] == 1] = self.real_G[permute2][self.mask_G[permute2] == 1][:, :, (0, 2, 1), :, :]
            #self.real_G[permute3][self.mask_G[permute3] == 1] = self.real_G[permute3][self.mask_G[permute3] == 1][:, :, (2, 1, 0), :, :]
            self.real_G[permute1] = torch.where(self.mask_G[permute1] == 1, self.real_G[permute1][:,  (1, 2, 0), :, :], self.real_G[permute1])
            self.real_G[permute2] = torch.where(self.mask_G[permute2] == 1, self.real_G[permute2][:,  (0, 2, 1), :, :], self.real_G[permute2])
            self.real_G[permute3] = torch.where(self.mask_G[permute3] == 1, self.real_G[permute3][:,  (2, 1, 0), :, :], self.real_G[permute3])
            #self.real_G_ref[permute1][self.mask_G[permute1].flip([1]) == 1] = self.real_G_ref[permute1][self.mask_G[permute1].flip([1]) == 1][:, :, (1, 2, 0), :, :]
            #self.real_G_ref[permute2][self.mask_G[permute2].flip([1]) == 1] = self.real_G_ref[permute2][self.mask_G[permute2].flip([1]) == 1][:, :, (0, 2, 1), :, :]
            #self.real_G_ref[permute3][self.mask_G[permute3].flip([1]) == 1] = self.real_G_ref[permute3][self.mask_G[permute3].flip([1]) == 1][:, :, (2, 1, 0), :, :]
            self.real_G_ref[permute1.flip(1)] = torch.where(self.mask_G[permute1.flip(1)] == 1, self.real_G_ref[permute1.flip(1)][:, (1, 2, 0), :, :], self.real_G_ref[permute1.flip(1)])
            self.real_G_ref[permute2.flip(1)] = torch.where(self.mask_G[permute2.flip(1)] == 1, self.real_G_ref[permute2.flip(1)][:, (0, 2, 1), :, :], self.real_G_ref[permute2.flip(1)])
            self.real_G_ref[permute3.flip(1)] = torch.where(self.mask_G[permute3.flip(1)] == 1, self.real_G_ref[permute3.flip(1)][:, (2, 1, 0), :, :], self.real_G_ref[permute3.flip(1)])

            permute1 = torch.randint(10, (N2,)) >= 5
            permute2 = torch.randint(10, (N2,)) >= 5
            permute3 = torch.randint(10, (N2,)) >= 5
            #self.real_D[permute1][self.mask_D[permute1] == 1] = self.real_D[permute1][self.mask_D[permute1] == 1][:, (1, 2, 0), :, :]
            #self.real_D[permute2][self.mask_D[permute2] == 1] = self.real_D[permute2][self.mask_D[permute2] == 1][:, (0, 2, 1), :, :]
            #self.real_D[permute3][self.mask_D[permute3] == 1] = self.real_D[permute3][self.mask_D[permute3] == 1][:, (2, 1, 0), :, :]
            self.real_D[permute1] = torch.where(self.mask_D[permute1] == 1, self.real_D[permute1][:, (1, 2, 0), :, :], self.real_D[permute1])
            self.real_D[permute2] = torch.where(self.mask_D[permute2] == 1, self.real_D[permute2][:, (1, 2, 0), :, :], self.real_D[permute2])
            self.real_D[permute3] = torch.where(self.mask_D[permute3] == 1, self.real_D[permute3][:, (1, 2, 0), :, :], self.real_D[permute3])



    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.background:
            if self.opt.attention:
                self.fake_G, self.mask_G = self.netG(self.real_G, mask_out=True)  # G(A)
            else:
                self.fake_G = self.netG(self.real_G, mask_in=self.mask_G)  # G(A)
        else:
            self.fake_G = self.netG(self.real_G)  # G(A)

        # Recreate original images from the fake images
        if self.opt.background and not self.opt.attention:
            self.rec_G_1 = self.netG(self.fake_G, mask_in=self.mask_G)   # G(G(A))
        else:
            self.rec_G_1 = self.netG(self.fake_G)  # G(G(A))

        # Recreate original images from fake image and real image as second
        if self.input_mode == 'mix':
            if self.opt.background and not self.opt.attention:
                self.rec_G_2A = self.netG(
                    torch.cat((self.fake_G[:, self.A, :, :, :].unsqueeze(1), self.real_G[:, self.A, :, :, :].unsqueeze(1)), dim=1),
                    mask_in=self.mask_G[:, self.A, :, :, :], mode=self.A)  # G(G(A))
                self.rec_G_2B = self.netG(
                    torch.cat((self.real_G[:, self.B, :, :, :].unsqueeze(1), self.fake_G[:, self.B, :, :, :].unsqueeze(1)), dim=1),
                    mask_in=self.mask_G[:, self.B, :, :, :], mode=self.B)  # G(G(A))
            else:
                self.rec_G_2A = self.netG(
                    torch.cat((self.fake_G[:, self.A, :, :, :].unsqueeze(1), self.real_G[:, self.A, :, :, :].unsqueeze(1)), dim=1),
                    mode=self.A)  # G(G(A))
                self.rec_G_2B = self.netG(
                    torch.cat((self.real_G[:, self.B, :, :, :].unsqueeze(1), self.fake_G[:, self.B, :, :, :].unsqueeze(1)), dim=1),
                    mode=self.B)  # G(G(A))

        # Use identity constraint for same image in both inputs
        # if self.input_mode == 'reflection'
        #     if self.opt.lambda_Identity > 0:
        #         N, _, C, H, W = self.real_G.shape
        #         self.iden_G = self.netG(
        #             self.real_G.reshape(2*N, 1, C, H, W).expand(2*N, 2, C, H, W),
        #             mask_in=self.mask_G.reshape(2*N, C, H, W),
        #             mode=self.A
        #         )

    def backward_D(self):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        _, _, C, H, W = self.fake_G.shape

        # Real
        pred_real = self.netD(self.real_D)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        if False: #self.opt.use_pool:
            fake_D = self.fake_pool.query(self.fake_G.reshape(-1, C, H, W))
        else:
            fake_D = self.fake_G.reshape(-1, C, H, W)
        pred_fake = self.netD(fake_D.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_G(self):
        """Calculate the loss for generator G"""
        _, _, C, H, W = self.fake_G.shape

        # GAN loss D(G(A))
        self.loss_GG = self.criterionGAN(self.netD(self.fake_G.reshape(-1, C, H, W)), True)

        # GAN Background loss
        if self.opt.background:
            self.loss_G_background = self.criterionBackground(
                self.real_G.reshape(-1, C, H, W) * (1 - self.mask_G.reshape(-1, C, H, W)),
                self.fake_G.reshape(-1, C, H, W) * (1 - self.mask_G.reshape(-1, C, H, W))) \
                                     * self.opt.lambda_Background
        else:
            self.loss_G_background = 0

        # Forward cycle loss || G(G(A)) - A||
        if self.input_mode == 'mix':
            self.loss_cycle_1 = self.criterionCycle(self.rec_G_1, self.real_G) * self.opt.lambda_G1
            self.loss_cycle_2A = self.criterionCycle(self.rec_G_2A, self.real_G[:, self.A, :, :, :]) * self.opt.lambda_G2
            self.loss_cycle_2B = self.criterionCycle(self.rec_G_2B, self.real_G[:, self.B, :, :, :]) * self.opt.lambda_G2

        # Identity
        if self.input_mode == 'reflection':
            if self.opt.lambda_Identity > 0:
                self.loss_identity = self.criterionCycle(self.real_G_ref, self.fake_G) * self.opt.lambda_Identity

        # combined loss and calculate gradients
        if self.input_mode == 'mix':
            self.loss_G = self.loss_GG + self.loss_G_background + self.loss_cycle_1 + self.loss_cycle_2A + self.loss_cycle_2B
        elif self.input_mode == 'reflection':
            self.loss_G = self.loss_GG + self.loss_G_background + self.loss_identity
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G
        self.set_requires_grad([self.netD], False)  # D require no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G gradients to zero
        self.backward_G()             # calculate gradients for G
        self.optimizer_G.step()       # update G weights
        # D
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()   # set D gradients to zero
        self.backward_D()      # calculate gradients for D
        self.optimizer_D.step()  # update D weights

    def runG(self, model_test_input=None):
        # Prepare data
        if model_test_input is None:
            real_G = self.real_G
            mask_G = self.mask_G
            N, _, C, H, W = real_G.shape
        else:
            real_G = model_test_input['real_G']
            N, C, H, W = real_G.shape
            if self.opt.background:
                if not self.opt.attention:
                    mask_G = model_test_input['mask_G'].unsqueeze(1).expand_as(real_G).reshape(-1, 2, C, H, W).to(self.device)

            real_G = real_G.reshape(-1, 2, C, H, W).to(self.device)
            N = N//2

        # Get results
        test_results = self.netG(real_G, mask_in=mask_G)

        iden_results = self.netG(
            real_G.reshape(2*N, 1, C, H, W).expand(2*N, 2, C, H, W),
            mask_in=mask_G.reshape(2*N, C, H, W),
            mode=self.A
        )

        return test_results, iden_results
