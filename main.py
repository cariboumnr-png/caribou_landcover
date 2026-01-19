'''Main entry point.'''

# import sys
# sys.stdout = open("output.txt", "w")

# third-party imports
import hydra
import omegaconf
# local imports
import src.dataset
import src.models
import src.training.factory

# add resolvers to omegaconf for conveniences
omegaconf.OmegaConf.register_new_resolver('c', lambda x: int(x * 100))

@hydra.main(config_path='./configs', config_name='main', version_base='1.3')
def test(config: omegaconf.DictConfig):
    '''Test function.'''
    data_sum = src.dataset.prepare_data(config.dataset)
    print(data_sum)

@hydra.main(config_path='./configs', config_name='main', version_base='1.3')
def main(config: omegaconf.DictConfig):
    '''doc'''

    # data preparation
    data_summary = src.dataset.prepare_data(config.dataset)
    print(data_summary)

    # setup multihead model
    model = src.models.multihead_unet(data_summary, config.models)

    # setup trainer
    trainer = src.training.factory.build_trainer(model, data_summary, config.trainer)
    trainer.set_head_state()
    trainer.train_one_epoch(1)
    print(trainer.state)

    # # setup curriculum
    # controller = src.training.factory.build_controller(trainer, config.curriculum)

    # # train
    # controller.fit()
#
if __name__ == '__main__':
    # test() # pylint: disable=no-value-for-parameter
    main() # pylint: disable=no-value-for-parameter
