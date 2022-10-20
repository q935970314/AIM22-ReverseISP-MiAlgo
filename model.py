import torch.nn as nn
import networks as N


class LiteISPNet_s(nn.Module):
    def __init__(self):
        super(LiteISPNet_s, self).__init__()
        ch_1 = 32
        ch_2 = 64
        ch_3 = 128
        n_blocks = 4

        self.head = N.seq(
            N.conv(in_channels=3, out_channels=ch_1, kernel_size=3, stride=2, padding=1, mode='C')
        )  # shape: (N, 3, H, W) -> (N, ch_1, H/2, W/2)

        self.down1 = N.seq(
            N.conv(ch_1, ch_1, mode='C'),
            N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
            N.conv(ch_1, ch_1, mode='C'),
            N.DWTForward(ch_1)
        )  # shape: (N, ch_1*4, H/4, W/4)

        self.down2 = N.seq(
            N.conv(ch_1*4, ch_1, mode='C'),
            N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
            N.DWTForward(ch_1)
        )  # shape: (N, ch_1*4, H/8, W/8)

        self.down3 = N.seq(
            N.conv(ch_1*4, ch_2, mode='C'),
            N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
            N.DWTForward(ch_2)
        )  # shape: (N, ch_2*4, H/16, W/16)

        self.middle = N.seq(
            N.conv(ch_2*4, ch_3, mode='C'),
            N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
            N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
            N.conv(ch_3, ch_2*4, mode='C')
        )  # shape: (N, ch_2*4, H/16, W/16)

        self.up3 = N.seq(
            N.DWTInverse(ch_2*4),
            N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
            N.conv(ch_2, ch_1*4, mode='C')
        )  # shape: (N, ch_1*4, H/8, W/8)

        self.up2 = N.seq(
            N.DWTInverse(ch_1*4),
            N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
            N.conv(ch_1, ch_1*4, mode='C')
        )  # shape: (N, ch_1*4, H/4, W/4)

        self.up1 = N.seq(
            N.DWTInverse(ch_1*4),
            N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
            N.conv(ch_1, ch_1, mode='C')
        )  # shape: (N, ch_1, H/2, W/2)

        self.tail = N.seq(
            N.conv(ch_1, 4, mode='C')
        )  # shape: (N, 4, H/2, W/2)

        self.tail_ca = N.seq(
            N.CALayer(4, 4)
        )  # shape: (N, 4, H/2, W/2)

        self.shading = N.seq(
            N.conv(ch_1, ch_1, mode='CR'),
            N.conv(ch_1, 1, mode='C'),
            nn.Sigmoid()
        )

    def forward(self, rgb):
        h = self.head(rgb)
        d1 = self.down1(h)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        m = self.middle(d3) + d3
        u3 = self.up3(m) + d2
        u2 = self.up2(u3) + d1
        u1 = self.up1(u2)
        out = self.tail(u1)
        out = self.tail_ca(out)
        shading_mask = self.shading(u1)
        return out * shading_mask