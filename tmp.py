class Generator(nn.Module):
    def __init__(self, output_dim, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(Generator, self).__init__()
        self.convlstm = CLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first, bias, return_all_layers)


        def downsample(in_feat, out_feat, normalize=True, dropout=True):
            layers = [nn.Conv2d(in_feat, out_feat, 3, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            if dropout:
                layers.append(nn.Dropout2d(0.5))
            return layers

        def upsample(in_feat, out_feat, normalize=True, dropout=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 3, stride=2, padding=1, output_padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            if dropout:
                layers.append(nn.Dropout2d(0.5))
            layers.append(nn.ReLU())
            return layers
        
        self.model = nn.Sequential(
            self.convlstm,
            *downsample(hidden_dim[-1], 16, normalize=False), # ->300
            *downsample(16, 32), # ->150
            *downsample(32, 32), # ->75
            nn.Conv2d(64, 64, 1), # -> 75
            *upsample(64, 32), # -> 8
            *upsample(32, 16), # -> 16
            *upsample(16, 8), # -> 32
            nn.Conv2d(8, output_dim, 3, 1, 1), # -> 128
            nn.Tanh()
        )

    def forward(self, x):
        """
        x : [b, t, c, h, w]
        """
        _, last_states = self.model[0](x)
        h = last_states[0][0] # [b, hidden_dim, h, w]
        output = self.model[:11](h)
        noise = torch.rand(x.shape[0], 32, 75, 75)
        output = torch.cat([output, noise], dim=1)
        output = self.model[11:](output).unsqueeze(1) # [b, 1, 1, h, w], 为了便于后续seq_len维度上concat

        return output