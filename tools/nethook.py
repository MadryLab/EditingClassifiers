'''
Utilities for instrumenting a torch model.

InstrumentedModel will wrap a pytorch model and allow hooking
arbitrary layers to monitor or modify their output directly.

nethook.run will run a pytorch model while hoooking it for just
the single invocation.
'''

import torch, numpy, types, copy, inspect, types, contextlib
from collections import OrderedDict, defaultdict

class Trace: # (contextlib.AbstractContextManager): for python 3.6+
    '''
    To retain the output of the named layer during the computation of
    the given network:

        with Trace(net, 'layer.name') as ret:
            _ = net(inp)
            representation = ret.output

    A layer module can be passed directly without a layer name, and
    its output will be retained.  By default, a direct reference to
    the output object is returned, but options can control this:

        clone=True  - retains a copy of the output, which can be
            useful if you want to see the output before it might
            be modified by the network in-place later.
        detach=True - retains a detached reference or copy.  (By
            default the value would be left attached to the graph.)
        retain_grad=True - request gradient to be retained on the
            output.  After backward(), ret.output.grad is populated.

    '''
    def __init__(self, module, layer=None, retain_output=True,
            retain_input=False, clone=False, detach=False, retain_grad=False,
            stop=False):
        '''
        Method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        '''
        retainer = self
        if layer is not None:
            module = get_module(module, layer)
        def retain_hook(m, inputs, output):
            if retain_input:
                retainer.input = recursive_copy(
                   inputs[0] if len(inputs) == 1 else inputs,
                   clone=clone,
                   detach=detach,
                   retain_grad=False) # retain_grad applies to output only.
            if retain_output:
                retainer.output = recursive_copy(
                    output,
                    clone=clone,
                    detach=detach,
                    retain_grad=retain_grad)
                # When retain_grad is set, also insert a trivial
                # copy operation.  That allows in-place operations
                # to follow without error.
                if retain_grad:
                     output = recursive_copy(output, clone=True, detach=False)
            if stop:
                raise StopForward()
            return output
        self.registered_hook = module.register_forward_hook(retain_hook)
        self.stop = stop

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        self.registered_hook.remove()

class StopForward(Exception):
    '''
    If the only output needed from running a network is the retained
    submodule then Trace(submodule, stop=True) will stop execution
    immediately after the retained submodule by raising the StopForward()
    exception.  When Trace is used as context manager, it catches that
    exception and can be used as follows:

    with Trace(net, layername, stop=True) as tr:
        net(inp) # Only runs the network up to layername
    print(tr.output)
    '''
    pass



'''
run will temporarily instrument a pytorch model for a single
invokation at a time, for example:

    ```
    output, layer1, layer2 = nethook.run(model, x)('layer1 layer2')
    ```

See InstrumentedRunnner for details on what can be hooked by run.
'''
def run(model, *args, **kwargs):
    return InstrumentedRunner(model, *args, **kwargs)

'''
InstrumentedRunner is used to invoke a model just once under
instrumentation.  Calling a model using a runner allows you
to specify the following:

    layers: a string layer name (like 'features.conv5_3'),
        or a list of such names or a space-separated list of names.
    edit: function for modifying the output of the layers
        or a {layername: func} dictionary of such functions.
    detach: False to return the tensor within the computation graph
    retain_grad: True to request gradient on the retained tensor
        next time backward() is called.

The output of the specified layers are teturn along with the
output of the whole model, in a list.
'''
class InstrumentedRunner:
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.args = args
        self.kwargs = kwargs
    def __call__(self, layers,
            edit=None, clone=None, detach=None, retain_grad=None):
        layers = split_retained_layers(layers)
        edit = split_edit_rules(layers, edit)
        with InstrumentedModel(self.model) as inst:
            inst.retain_layers(layers,
                    clone=clone, detach=detach, retain_grad=retain_grad)
            if edit:
                for layer, rule in edit.items():
                    inst.edit_layer(layer, rule=rule)
            result = [inst(*self.args, **self.kwargs)]
            for layer in layers:
                result.append(inst.retained_layer(layer))
        return result

class InstrumentedModel(torch.nn.Module):
    '''
    A wrapper for hooking, probing and intervening in pytorch Modules.
    Example usage:

    ```
    model = load_my_model()
    with InstrumentedModel(model) as inst:
        inst.retain_layer(layername)
        inst.edit_layer(layername, ablation=0.5, replacement=target_features)
        inst(inputs)
        original_features = inst.retained_layer(layername)
    ```
    '''

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._retained = OrderedDict()
        self._clone_retained = {}
        self._detach_retained = {}
        self._retain_grad = {}
        self._editargs = defaultdict(dict)
        self._editrule = {}
        self._hooked_layer = {}
        self._old_forward = {}
        if isinstance(model, torch.nn.Sequential):
            self._hook_sequential()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def layer_names(self):
        '''
        Returns a list of layer names.
        '''
        return [name for name, _ in self.model.named_modules()]

    def retain_layer(self, layername, detach=None, retain_grad=None, clone=None):
        '''
        Pass a fully-qualified layer name (E.g., module.submodule.conv3)
        to hook that layer and retain its output each time the model is run.
        A pair (layername, aka) can be provided, and the aka will be used
        as the key for the retained value instead of the layername.
        '''
        self.retain_layers([layername],
                detach=detach, retain_grad=retain_grad, clone=clone)

    def retain_layers(self, layernames, detach=None, retain_grad=None, clone=None):
        '''
        Retains a list of a layers at once.
        '''
        if retain_grad:
            assert not detach
            detach = False
        if retain_grad is None:
            detach = False
        if detach is None:
            detach = True
        self.add_hooks(layernames)
        for layername in layernames:
            aka = layername
            if not isinstance(aka, str):
                layername, aka = layername
            if aka not in self._retained:
                self._retained[aka] = None
                self._clone_retained[aka] = clone
                self._detach_retained[aka] = detach
                self._retain_grad[aka] = retain_grad

    def stop_retaining_layers(self, layernames):
        '''
        Removes a list of layers from the set retained.
        '''
        self.add_hooks(layernames)
        for layername in layernames:
            aka = layername
            if not isinstance(aka, str):
                layername, aka = layername
            if aka in self._retained:
                del self._retained[aka]
                del self._clone_retained[aka]
                del self._detach_retained[aka]
                del self._retain_grad[aka]

    def retained_features(self, clear=False):
        '''
        Returns a dict of all currently retained features.
        '''
        result = OrderedDict(self._retained)
        if clear:
            for k in result:
                self._retained[k] = None
        return result

    def retained_layer(self, aka=None, clear=False):
        '''
        Retrieve retained data that was previously hooked by retain_layer.
        Call this after the model is run.  If clear is set, then the
        retained value will return and also cleared.
        '''
        if aka is None:
            # Default to the first retained layer.
            aka = next(self._retained.keys().__iter__())
        result = self._retained[aka]
        if clear:
            self._retained[aka] = None
        return result

    def edit_layer(self, layername, rule=None, **kwargs):
        '''
        Pass a fully-qualified layer name (E.g., module.submodule.conv3)
        to hook that layer and modify its output each time the model is run.
        The output of the layer will be modified to be a convex combination
        of the replacement and x interpolated according to the ablation, i.e.:
        `output = x * (1 - a) + (r * a)`.
        '''
        if not isinstance(layername, str):
            layername, aka = layername
        else:
            aka = layername

        # The default editing rule is apply_ablation_replacement
        if rule is None:
            rule = apply_ablation_replacement

        self.add_hooks([(layername, aka)])
        self._editargs[aka].update(kwargs)
        self._editrule[aka] = rule

    def remove_edits(self, layername=None):
        '''
        Removes edits at the specified layer, or removes edits at all layers
        if no layer name is specified.
        '''
        if layername is None:
            self._editargs.clear()
            self._editrule.clear()
            return

        if not isinstance(layername, str):
            layername, aka = layername
        else:
            aka = layername
        if aka in self._editargs:
            del self._editargs[aka]
        if aka in self._editrule:
            del self._editrule[aka]

    def add_hooks(self, layernames):
        '''
        Sets up a set of layers to be hooked.

        Usually not called directly: use edit_layer or retain_layer instead.
        '''
        needed = set()
        aka_map = {}
        for name in layernames:
            aka = name
            if not isinstance(aka, str):
                name, aka = name
            if self._hooked_layer.get(aka, None) != name:
                aka_map[name] = aka
                needed.add(name)
        if not needed:
            return
        for name, layer in self.model.named_modules():
            if name in aka_map:
                needed.remove(name)
                aka = aka_map[name]
                self._hook_layer(layer, name, aka)
        for name in needed:
            raise ValueError('Layer %s not found in model' % name)

    def _hook_layer(self, layer, layername, aka):
        '''
        Internal method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        '''
        if aka in self._hooked_layer:
            raise ValueError('Layer %s already hooked' % aka)
        if layername in self._old_forward:
            raise ValueError('Layer %s already hooked' % layername)
        self._hooked_layer[aka] = layername
        self._old_forward[layername] = (layer, aka,
                layer.__dict__.get('forward', None))
        editor = self
        original_forward = layer.forward
        def new_forward(self, *inputs, **kwargs):
            original_x = original_forward(*inputs, **kwargs)
            x = editor._postprocess_forward(original_x, aka)
            return x
        layer.forward = types.MethodType(new_forward, layer)

    def _unhook_layer(self, aka):
        '''
        Internal method to remove a hook, restoring the original forward method.
        '''
        if aka not in self._hooked_layer:
            return
        layername = self._hooked_layer[aka]
        # Remove any retained data and any edit rules
        if aka in self._retained:
            del self._retained[aka]
            del self._clone_retained[aka]
            del self._detach_retained[aka]
            del self._retain_grad[aka]
        self.remove_edits(aka)
        # Restore the unhooked method for the layer
        layer, check, old_forward = self._old_forward[layername]
        assert check == aka
        if old_forward is None:
            if 'forward' in layer.__dict__:
                del layer.__dict__['forward']
        else:
            layer.forward = old_forward
        del self._old_forward[layername]
        del self._hooked_layer[aka]

    def _postprocess_forward(self, x, aka):
        '''
        The internal method called by the hooked layers after they are run.
        '''
        # Retain output before edits, if desired.
        if aka in self._retained:
            self._retained[aka] = recursive_copy(x,
                    clone=self._clone_retained[aka],
                    detach=self._detach_retained[aka],
                    retain_grad=self._retain_grad[aka])
            if self._retain_grad[aka]:
                # When retain_grad is set, also insert a trivial
                # copy operation.  That allows in-place operations to follow
                # without error.
                x = recursive_copy(x, clone=True, detach=False)
        # Apply any edits requested.
        rule = self._editrule.get(aka, None)
        if rule is not None:
            x = invoke_with_optional_args(
                rule, x, self, name=aka, **(self._editargs[aka]))
        return x

    def _hook_sequential(self):
        '''
        Replaces 'forward' of sequential with a version that takes
        additional keyword arguments: layer allows a single layer to be run;
        first_layer and last_layer allow a subsequence of layers to be run.
        '''
        model = self.model
        self._hooked_layer['.'] = '.'
        self._old_forward['.'] = (model, '.',
                model.__dict__.get('forward', None))
        def new_forward(this, x, layer=None, first_layer=None, last_layer=None):
            # TODO: decide whether to support hierarchical names here.
            assert layer is None or (first_layer is None and last_layer is None)
            first_layer, last_layer = [str(layer) if layer is not None
                    else str(d) if d is not None else None
                    for d in [first_layer, last_layer]]
            including_children = (first_layer is None)
            for name, layer in this._modules.items():
                if name == first_layer:
                    first_layer = None
                    including_children = True
                if including_children:
                    x = layer(x)
                if name == last_layer:
                    last_layer = None
                    including_children = False
            assert first_layer is None, '%s not found' % first_layer
            assert last_layer is None, '%s not found' % last_layer
            return x
        model.forward = types.MethodType(new_forward, model)

    def close(self):
        '''
        Unhooks all hooked layers in the model.
        '''
        for aka in list(self._old_forward.keys()):
            self._unhook_layer(aka)
        assert len(self._old_forward) == 0

def recursive_copy(x, clone=None, detach=None, retain_grad=None):
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v) for v in x])
    else:
        assert False, f'Unknown type {type(x)} cannot be broken into tensors.'

def apply_ablation_replacement(x, imodel, **buffers):
    if buffers is not None:
        # Apply any edits requested.
        a = make_matching_tensor(buffers, 'ablation', x)
        if a is not None:
            x = x * (1 - a)
            v = make_matching_tensor(buffers, 'replacement', x)
            if v is not None:
                x += (v * a)
    return x

def make_matching_tensor(valuedict, name, data):
    '''
    Converts `valuedict[name]` to be a tensor with the same dtype, device,
    and dimension count as `data`, and caches the converted tensor.
    '''
    v = valuedict.get(name, None)
    if v is None:
        return None
    if not isinstance(v, torch.Tensor):
        # Accept non-torch data.
        v = torch.from_numpy(numpy.array(v))
        valuedict[name] = v
    if not v.device == data.device or not v.dtype == data.dtype:
        # Ensure device and type matches.
        assert not v.requires_grad, '%s wrong device or type' % (name)
        v = v.to(device=data.device, dtype=data.dtype)
        valuedict[name] = v
    if len(v.shape) < len(data.shape):
        # Ensure dimensions are unsqueezed as needed.
        assert not v.requires_grad, '%s wrong dimensions' % (name)
        v = v.view((1,) + tuple(v.shape) +
                (1,) * (len(data.shape) - len(v.shape) - 1))
        valuedict[name] = v
    return v

def subsequence(sequential, first_layer=None, last_layer=None,
        after_layer=None, upto_layer=None, single_layer=None,
        share_weights=False):
    '''
    Creates a subsequence of a pytorch Sequential model, copying over
    modules together with parameters for the subsequence.  Only
    modules from first_layer to last_layer (inclusive) are included,
    or modules between after_layer and upto_layer (exclusive).
    Handles descent into dotted layer names as long as all references
    are within nested Sequential models.

    If share_weights is True, then references the original modules
    and their parameters without copying them.  Otherwise, by default,
    makes a separate brand-new copy.
    '''
    assert ((single_layer is None) or
            (first_layer is last_layer is after_layer is upto_layer is None))
    if single_layer is not None:
        first_layer = single_layer
        last_layer = single_layer
    first, last, after, upto = [None if d is None else d.split('.')
            for d in [first_layer, last_layer, after_layer, upto_layer]]
    return hierarchical_subsequence(sequential, first=first, last=last,
            after=after, upto=upto, share_weights=share_weights)

def hierarchical_subsequence(sequential, first, last, after, upto,
        share_weights=False, depth=0):
    '''
    Recursive helper for subsequence() to support descent into dotted
    layer names.  In this helper, first, last, after, and upto are
    arrays of names resulting from splitting on dots.  Can only
    descend into nested Sequentials.
    '''
    assert (last is None) or (upto is None)
    assert (first is None) or (after is None)
    if first is last is after is upto is None:
        return sequential if share_weights else copy.deepcopy(sequential)
    #assert isinstance(sequential, torch.nn.Sequential), ('.'.join(
    #    (first or last or after or upto)[:depth] or 'arg') + ' not Sequential')
    including_children = (first is None) and (after is None)
    included_children = OrderedDict()
    (F, FN), (L, LN), (A, AN), (U, UN) = [
            (d[depth], (None if len(d) == depth+1 else d))
            if d is not None else (None, None)
            for d in [first, last, after, upto]]
    for name, layer in sequential._modules.items():
        if name == F:
            first = None
            including_children = True
        if name == A and AN is not None:
            after = None
            including_children = True
        if name == U and UN is None:
            upto = None
            including_children = False
        if including_children:
            FR, LR, AR, UR = [n if n is None or n[depth] == name else None
                    for n in [FN, LN, AN, UN]]
            chosen = hierarchical_subsequence(layer,
                    first=FR, last=LR, after=AR, upto=UR,
                    share_weights=share_weights, depth=depth+1)
            if chosen is not None:
                included_children[name] = chosen
        if name == L:
            last = None
            including_children = False
        if name == U and UN is not None:
            upto = None
            including_children = False
        if name == A and AN is None:
            after = None
            including_children = True
    for name in [first, last, after, upto]:
        if name is not None:
            raise ValueError('Layer %s not found' % '.'.join(name))
    # Omit empty subsequences except at the outermost level,
    # where we should not return None.
    if not len(included_children) and depth > 0:
        return None
    result = torch.nn.Sequential(included_children)
    result.training = sequential.training
    return result

def set_requires_grad(requires_grad, *models):
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, 'unknown type %r' % type(model)

def get_module(model, name):
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)

def get_parameter(model, name):
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)

def replace_module(model, name, new_module):
    split_name = name.split('.')
    attr_name = split_name[-1]
    parent_name = '.'.join(split_name[:-1])
    parent = get_module(model, parent_name)
    # original_module = getattr(parent, attr_name)
    setattr(parent, attr_name, new_module)

def invoke_with_optional_args(fn, *args, **kwargs):
    argspec = inspect.getfullargspec(fn)
    kwtaken = 0
    if argspec.varkw is None:
        kwtaken = len([k for k in kwargs if k in argspec.args])
        kwargs = {k: v for k, v in kwargs.items()
                if k in argspec.args or
                argspec.kwonlyargs and k in argspec.kwonlyargs}
    if argspec.varargs is None:
        args = args[:len(argspec.args) - kwtaken]
    return fn(*args, **kwargs)

def split_retained_layers(s):
    if s is None:
        return []
    if isinstance(s, str):
        return s.split(' ')
    return s

def split_edit_rules(layers, d):
    if d is None:
        return d
    if isinstance(d, types.FunctionType):
        return {k: d for k in layers}
    result = {}
    for k, v in d.items():
        if ' ' in k:
            for kk in k.split(' '):
                result[kk] = v
        else:
            result[k] = v
    return result
