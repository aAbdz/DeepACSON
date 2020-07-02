# -*- coding: utf-8 -*-

save_path = 'ADDRESS: save_path'
preview_data_path = 'ADDRESS: preview image of .h5 format'
preview_kwargs    = {
    'export_class': [0,1,2,3],
    'max_z_pred': 1 
}

initial_prev_h = 1  # hours: time after which the first preview is made
prev_save_h = 1  # hours: time interval between planned previews.
data_class = 'BatchCreatorImage'
background_processes = 2

data_init_kwargs = {
    'd_path' : 'ADSRESS: TRAINING SET, RAW FILES',
    'l_path': 'ADDRESS: TRAINING SET, LABEL FILES',
    'd_files': [('raw%i.h5' %i, 'raw') for i in range(4)],
    'l_files': [('label%i.h5' %i, 'lab') for i in range(4)],
    'aniso_factor': 3,
    'valid_cubes': [2],
}

flip_data = True
data_batch_args = {
    'grey_augment_channels': [0],
    'warp': 0.5,
    'warp_args': {
        'sample_aniso': True,
        'perspective': True
    }
}

n_steps = 1000000
max_runtime = 7 * 24 * 3600 # in seconds
history_freq = 200
monitor_batch_size = 30
optimiser = 'Adam'
optimiser_params = {
    'lr': 0.00005,
    'mom': 0.9,
    'beta2': 0.999,
    'wd': 0.5e-5}

schedules = {
    'lr': {'dec': 0.995}, # decay (multiply) lr by this factor every 1000 steps
}

batch_size = 1

def create_model():
    from elektronn2 import neuromancer as nm
    in_sh = (None,1,23,185,185)
    inp = nm.Input(in_sh, 'b,f,z,x,y', name='raw')

    out   = nm.Conv(inp, 20,  (1,3,3))
    out   = nm.Conv(out, 40,  (1,3,3))
    out   = nm.Conv(out, 64,  (1,4,4), (1,2,2))
    out   = nm.Conv(out, 80,  (4,4,4), (2,1,1))

    out   = nm.Conv(out, 100, (3,4,4))
    out   = nm.Conv(out, 100, (3,4,4))
    out   = nm.Conv(out, 150, (2,4,4))
    out   = nm.Conv(out, 200, (1,4,4))
    out   = nm.Conv(out, 200, (1,4,4))

    out   = nm.Conv(out, 200, (1,1,1))
    out   = nm.Conv(out,   4, (1,1,1), activation_func='lin')
    probs = nm.Softmax(out)

    target = nm.Input_like(probs, override_f=1, name='target')
    loss_pix  = nm.MultinoulliNLL(probs, target, target_is_sparse=True, class_weights=[0.2,0.5,1,1])

    loss = nm.AggregateLoss(loss_pix , name='loss')
    errors = nm.Errors(probs, target, target_is_sparse=True)

    model = nm.model_manager.getmodel()
    model.designate_nodes(
        input_node=inp,
        target_node=target,
        loss_node=loss,
        prediction_node=probs,
        prediction_ext=[loss, errors, probs]
    )
    return model


if __name__ == '__main__':

    import traceback

    model = create_model()

    try:
        model.test_run_prediction()
    except Exception as e:
        traceback.print_exc()
        print('Test run failed.\nIn case your GPU ran out of memory, the '
              'principal setup might still be working')

