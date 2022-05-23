"""Register deep model"""
import os, logging, argparse
from azureml.core import Datastore, Run, Dataset
from azureml.core.model import Model

logger = logging.getLogger(__name__)

PIPELINE_URL = "https://msasg.visualstudio.com/Bing_and_IPG/_build?definitionId=21335&_a=summary"


def parse_args():
    """
        Parse arguments
    """
    parser = argparse.ArgumentParser()

    # Resource parameters
    parser.add_argument("--model_path", type=str, default=None, help="Relative path to model")
    parser.add_argument("--model_name", type=str, default=None, help="Model name")
    parser.add_argument("--model_description", type=str, default=None, help="Model description")
    parser.add_argument("--aether_exp", type=str, default=None, help="Aether Exp Id")
    parser.add_argument("--datastore", type=str, default=None, help="Data store name")
    parser.add_argument("--output", type=str, default=None, help="The output path")

    args, _ = parser.parse_known_args()
    return args

def get_datastore(ws, datastore_name):
    """
        Get datastore with given [datastore_name] from workspace [ws]
    """

    try:
        datastore = Datastore.get(ws, datastore_name)
    except:
        print("Exception occured during get pipeline data store", datastore_name)
        raise Exception("datastore %s not initialized" % (datastore_name,))
        
    return datastore

def output_results(target_folder, model_name, model_version, model_description):
    """
        Write output to [target_folder]/output
    """
    if target_folder:
        os.makedirs(target_folder, exist_ok=True)

        output_lines = [
            f"Model name:               {model_name}",
            f"Model version:            {model_version}",
            f"Model description:        {model_description}",
            f"AutoDeploy pipeline URL:  {PIPELINE_URL}"
        ]
        with open(os.path.join(target_folder, "output"), "w") as f:
            f.write("\n".join(output_lines))

def main(args):
    """
        The main function to register a model.
    """
    run = Run.get_context()
    ws = run.experiment.workspace
    datastore = Datastore(ws, args.datastore)
    model_ds = Dataset.File.from_files(path=[(datastore, args.model_path)])

    with model_ds.mount() as model_mount:
        real_model_path = model_mount.mount_point

        # Register model
        model = Model.register(ws, 
                    model_name=args.model_name, 
                    model_path=real_model_path, 
                    description=args.model_description, 
                    tags={"aml_job_id": run.get_details()["runId"], "aether_exp_id": args.aether_exp})

        logger.info('Name: %s', model.name)
        logger.info('Version: %s', model.version)
        logger.info('Description: %s', args.model_description)

        output_results(args.output, model.name, model.version, args.model_description)

if __name__ == "__main__":
    args = parse_args()
    logger.info("args: %s", args)

    main(args)