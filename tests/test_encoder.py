import unittest

import torch

from src.models.encoder import KeystrokeEncoder, SiameseNetwork
from src.models.loss import ContrastiveLoss


class EncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.encoder = KeystrokeEncoder(dropout=0)

    def test_shape_norm_and_padding_independence(self):
        self.encoder.eval()
        prefix = torch.randn(1, 11, 3)
        with torch.no_grad():
            reference = self.encoder(prefix, torch.tensor([11]))
            padded = torch.cat((prefix, torch.full((1, 59, 3), float('nan'))), dim=1)
            output = self.encoder(padded, torch.tensor([11]))
            torch.testing.assert_close(output, reference, atol=1e-6, rtol=1e-5)
            lengths = torch.tensor([11, 70, 2])  # deliberately unsorted
            batch = torch.cat((padded, torch.randn(2, 70, 3)))
            embeddings = self.encoder(batch, lengths)
        self.assertEqual(tuple(embeddings.shape), (3, 64))
        torch.testing.assert_close(embeddings.norm(dim=1), torch.ones(3))
        torch.testing.assert_close(embeddings[0], reference[0], atol=1e-6, rtol=1e-5)

    def test_loss_convention_and_zero_distance_gradients(self):
        a = torch.tensor([[1., 0.], [1., 0.], [1., 0.]], requires_grad=True)
        b = torch.tensor([[1., 0.], [-1., 0.], [1., 0.]], requires_grad=True)
        loss = ContrastiveLoss()(a, b, torch.tensor([1., 0., 0.]))
        torch.testing.assert_close(loss, torch.tensor(1/3))
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue(torch.isfinite(b.grad).all())
        close = torch.tensor([[1., 0.]])
        far = torch.tensor([[-1., 0.]])
        self.assertGreater(ContrastiveLoss()(close, far, torch.ones(1)),
                           ContrastiveLoss()(close, close, torch.ones(1)))

    def test_shared_encoder_gradients_and_optimizer_step(self):
        model = SiameseNetwork(self.encoder)
        self.assertIs(model.encoder, self.encoder)
        a, b = torch.randn(8, 70, 3), torch.randn(8, 70, 3)
        lengths = torch.tensor([11, 70, 2, 25, 30, 5, 60, 40])
        labels = torch.tensor([1., 0.] * 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = self.encoder.conv1.weight.detach().clone()
        loss = ContrastiveLoss()(*model(a, lengths, b, lengths), labels)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        self.assertFalse(torch.equal(before, self.encoder.conv1.weight))

    def test_invalid_inputs(self):
        for lengths in (torch.tensor([0]), torch.tensor([71]), torch.tensor([1.5])):
            with self.assertRaises(ValueError):
                self.encoder(torch.zeros(1, 70, 3), lengths)
        with self.assertRaises(ValueError):
            ContrastiveLoss()(torch.ones(1, 64), torch.ones(1, 64), torch.tensor([2.]))
