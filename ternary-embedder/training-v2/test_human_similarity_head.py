import unittest
import torch
import torch.nn.functional as F
from fit_human_similarity_head import FeatureScales,ranking_loss


class HeadTests(unittest.TestCase):
    def test_human_rank_loss_ignores_label_scale(self):
        similarities=torch.tensor([.2,.7,.4],requires_grad=True)
        labels=torch.tensor([.1,.8,.3])
        self.assertTrue(torch.equal(ranking_loss(similarities,labels),ranking_loss(similarities,labels*3+2)))

    def test_scales_fold_into_existing_projection(self):
        torch.manual_seed(2)
        latent=torch.randn(7,11);projection=torch.randn(5,11);head=FeatureScales(5)
        with torch.no_grad():head.log_scales.copy_(torch.linspace(-.2,.2,5))
        expected=head(F.normalize(latent@projection.T,dim=-1))
        actual=F.normalize(latent@(head.scales()[:,None]*projection).T,dim=-1)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
        self.assertEqual(sum(p.numel() for p in head.parameters()),5)


if __name__=='__main__':unittest.main()
