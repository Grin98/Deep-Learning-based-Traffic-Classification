from training.experiment import Experiment


class FlowPicExperiment(Experiment):
    # TODO: Implement the training loop
    def __run__(self, bs_train=128, bs_test=None, batches=100, epochs=100, early_stopping=3, checkpoints=None, lr=1e-3,
                reg=1e-3, filters_per_layer=[64], layers_per_block=2, pool_every=2, hidden_dims=[1024], ycn=False,
                **kw):
        pass


if __name__ == '__main__':
    parsed_args = FlowPicExperiment.parse_cli()
    print(f'*** Starting {FlowPicExperiment.__name__} with config:\n{parsed_args}')
    exp = FlowPicExperiment(**vars(parsed_args))
