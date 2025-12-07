from experiments.baselines.ShuffleNetV2.ShuffleNetV2Experiment import ShuffleNetV2Experiment
from src.services.CamBaseService import *


def test_cam_service_shuffle():
    # set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # define minimal config required by the experiment class
    # image_size isn't strictly used by shufflenet class logic but good to have
    config = {
        'epochs': 1,
        'batch_size': 10,
        'image_size': 224
    }

    print("initializing experiment context...")
    # instantiate the experiment class (shufflenetv2)
    # this is needed because cam service uses experiment.create_model()
    experiment = ShuffleNetV2Experiment(config)
    experiment.dataset_name = rvf10k

    # define the model filename
    model_filename = "rvf10k_ShuffleNetV2_pytorch.pth"

    # check if model exists before running
    if not (MODEL_FOLDER / model_filename).exists():
        print(f"warning: model file {model_filename} not found in {MODEL_FOLDER}.")
        print("please run the shufflenet training script first.")
        return

    print(f"initializing cam service for {model_filename}...")
    # instantiate the cam service
    # we use layer_cam as requested
    cam_service = CamBaseService(
        experiment=experiment,
        model_name=model_filename,
        cam_method=CAM_TYPE.LAYER.value,
        image_mode=LOAD_MODE.ONLY_FAKE.value,
    )

    # set batch limit (how many images to process)
    cam_service.batch_limit = 10

    print("starting cam generation...")
    # run the service
    cam_service.run()