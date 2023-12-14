from ShapeChecker import ShapeCheck
import torch, math

class MetaRegFunction :
    @classmethod
    def get_requests(cls) :
        return {}

class EulerPottsl1(MetaRegFunction) :
     @classmethod
     def __call__(cls, sc, PredV, **kwargs) :
        eu_loss = torch.abs(PredV[:,:,:-1] - PredV[:,:,1:])
        return sc.reduce(eu_loss, 'b l tn i j -> b', 'mean')

class EulerPottsl1R3(MetaRegFunction) :
    @classmethod
    def __call__(cls, sc, PredV, FlowV, **kwargs) :
        eu_loss = sc.reduce(torch.abs(PredV[:,:,:-1] - PredV[:,:,1:]), 'b l tn i j -> b tn i j', 'mean')
        tempo_flow = sc.reduce(torch.abs(FlowV[:,:,:-1] - FlowV[:,:,1:]), 'b c tn i j -> b tn i j', 'sum')
        qt = torch.quantile(sc.rearrange(tempo_flow, 'b tn i j -> b (tn i j)'), 0.99, dim=1)[:, None, None, None]
        return sc.reduce(eu_loss * (tempo_flow < qt), 'b tn i j -> b', 'mean')

    @classmethod
    def get_requests(cls) :
        return {'Flow'}

class Potts(MetaRegFunction) :
    @classmethod
    def __call__(cls, sc, PredV, **kwargs) :
        eu_loss = sc.reduce(-(PredV[:,:,:-1] * PredV[:,:,1:]), 'b l tn i j -> b tn i j', 'sum')
        return sc.reduce(eu_loss, 'b tn i j -> b', 'mean')

class Entropy(MetaRegFunction) :
     @classmethod
     def __call__(cls, sc, PredV, **kwargs) :
        ents = -PredV*torch.log(torch.clamp(PredV, min=1e-4))
        ents = sc.reduce(ents, 'b l t i j -> b t i j', 'sum')
        return sc.reduce(ents, 'b t i j -> b', 'mean')

class Regularisation :
    def __init__(self) :
        self.regs = {'EulerPottsl1' : EulerPottsl1(),
                    'EulerPottsl1R3':EulerPottsl1R3(),
                    'Entropy' : Entropy(),
                    'Potts' : Potts()}
        self.sc = ShapeCheck([2], 'c')

    def loss(self, name, batch) :
        self.sc.update(batch['Theta'].shape, 'b l ft')
        self.sc.update(batch['PredV'].shape, 'b l t i j')
        self.sc.update(batch['FlowV'].shape, 'b c t i j')
        return self.regs[name](self.sc, **batch)
