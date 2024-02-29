Definition of component spec
==================================

This document describes the specification to define an AzureML component. The spec should be in YAML file format.


### Component Definition

| Name                | Type                                                     | Required | Description                                                  |
| ------------------- | -------------------------------------------------------- | -------- | ------------------------------------------------------------ |
| $schema             | String                                                   | Yes      | Specify the version of the schema of spec. Example: https://componentsdk.azureedge.net/jsonschema/CommandComponent.json |
| name                | String                                                   | Yes      | Name of the component. Name will be unique identifier of the component. |
| version             | String                                                   | Yes      | Version of the component. Could be arbitrary text, but it is recommended to follow the [Semantic Versioning](https://semver.org/) specification. |
| display_name        | String                                                   | No       | Display name of the component. Defaults to same as name. |
| type                | String                                                   | No       | Defines type of the component. Could be `CommandComponent`, `ParallelComponent`, etc.. Component type should match the $schema. |
| description         | String                                                   | No       | Detailed description of the Component. |
| tags                | Dictionary<String>                                       | No | A list of key-value pairs to describe the different perspectives of the component. Each tag's key and value should be one word or a short phrase, e.g. `Product:Office`, `Domain:NLP`, `Scenario:Image Classification`. |
| is_deterministic    | Boolean                                                  | No       | Specify whether the component will always generate the same result when given the same input data. For sweep component, default value is `False`, for other components, defaults to `True` if not specified. Typically for components which will load data from external resources, e.g. Import data from a given url, should set to `False` since the data to where the url points to may be updated. |
| successful_return_code| String                                                 | No       | Specify how command return code is interpreted. It only supports "Zero" and "ZeroOrGreater", default to "Zero" if not specified.|
| inputs              | Dictionary < String, [Input](#input) or [Parameter](#parameter) > | No       | Defines input ports and parameters of the component. The string key is the name of Input/Parameter, which should be a valid python variable name. |
| outputs             | Dictionary < String, [Output](#output) >                    | No       | Defines output ports of the component. The string key is the name of Output, which should be a valid python variable name. |
| code                | String                                                   | No       | Location of the [Code](#code) snapshot. |
| environment         | [Environment](#environment)                              | No       | An Environment defines the runtime environment for the component to run. Refer to [here](component-spec-topics/running-environment.md) for details. |
|environment_variables| Dictionary<String, String>                               | No       | Used to specify default environment variables to be passed. It is a dictionary of environment name to environment value mapping. User can use this to adjust some component runtime behavior which is not exposed as component parameter, e.g. enable some debug switch. Only a subset of component types support this like: Command, Distributed, Sweep, Parallel. |
| command             | String                                             | No    | Specify the command to start to run the component code.         |
| launcher | [Launcher](#launcher) | No | Launcher settings for `DistributedComponent` only. Refer to [here](#launcher) for details. |
| parallel | [Parallel](#parallel) | No | Settings for `ParallelComponent` only. Refer to [here](#parallel) for details. |
| hdinsight | [HDInsight](#hdinsight) | No | Settings for `HDInsightComponent` only. Refer to [here](#hdinsight-internal-only-for-now) for details. |
| scope | [Scope](#scope) | No | Settings for `ScopeComponent` only. Refer to [here](#scope) for details. |
| hemera | [Hemera](#hemera) | No | Settings for `HemeraComponent` only. Refer to [here](#hemera) for details. |
| starlite | [Starlite](#starlite) | No | Settings for `StarliteComponent` only. Refer to [here](#starlite) for details. |
| ae365exepool | [AE365ExePool](#ae365exepool) | No | Settings for `AE365ExePoolComponent` only. Refer to [here](#ae365exepool) for details. |
| aetherbridge | [AetherBridge](#aetherbridge) | No | Settings for `AetherBridgeComponent` only. Refer to [here](#aetherbridge) for details. |

### Name

Our recommendation to the component name will be something like `company`.`team`.`name-of-component`.
Current constraint for name: only accept letters, numbers and -._

Sample names:

```
microsoft.office.smart-compose
my-awesome-components.ner-bert
```

Builtin component name will be prefixed with `azureml://`.

Sample names:
```
azureml://Select Columns in Dataset
```

Note:
If you have a legacy module, you can load it using Component.load(name="{namespace}://{name}).

### Description

Please note if you write markdown in description, our portal UX will display a nicely formatted description.
For example:
```
description: |
  # A dummy training module
  - list 1
  - list 2
```

The above example use ***literal*** style with the indicator `|` to write multi-line in yaml.

See [reference](https://yaml-multiline.info/) for more details of the yaml multi-line format.

### Code

A code snapshot can be expressed as one of 3 things:

      1. a local file path relative to the file where it is referenced e.g. '../'. Register Only support this form now.
      2. an http url e.g. 'http://github.com/foo/bar/dir#239870234080' [not ready for use]
      3. a snapshot id, e.g.: aml://6560575d-fa06-4e7d-95fb-f962e74efd7a/azml-rg/sandbox-ws/snapshots/293lkw0j23fw8cv. [not ready for use]

See [reference](component-spec-topics/code-snapshot.md) for more details of code snapshot.

### Tags
Some convention tags used by azure-ml-component package. Refer to [get-started-train](https://github.com/Azure/DesignerPrivatePreviewFeatures/blob/master/azure-ml-components/samples/components/get-started-train/train.yaml)
and [get-started-score](https://github.com/Azure/DesignerPrivatePreviewFeatures/blob/master/azure-ml-components/samples/components/get-started-score/score.yaml) for more details. Follow [how to access instructions](../getting_started.html#how-to-access) if you meet 404 error when accessing the samples.

| Name        | Type         | Required | Description                                                  |
| ----------- | ------------ | -------- | ------------------------------------------------------------ |
| codegenBy | String | No  | The component spec might be generated by some automation tool. Set the tool name into this field. e.g. `dsl.component` |
| contact     | String       | No       | The contact info of the component's author. Typically contains user or organization's name and email. e.g. `AzureML Studio Team <stcamlstudiosg@microsoft.com>`. |
| helpDocument | String       | No       | The url of the component's documentation. The url is shown as a link on AzureML Designer's page. |

### Input

Defines an input port of the component. Refer to [here](component-spec-topics/inputs-and-outputs.md) for details.

| Name         | Type                    | Required | Description                                                  |
| ------------ | ----------------------- | -------- | ------------------------------------------------------------ |
| type         | String or  List<String> | Yes      | Defines the data type(s) of this input port. Refer to [Data Types for Inputs/Outputs](#data-types-for-inputs/outputs) for details. |
| optional     | Boolean                 | No       | Indicates whether this input is an optional port. Defaults to `False` if not specified. |
| description  | String                  | No       | Detailed description to the input port.      |
| is_resource  | Boolean                 | No       | Set to true to mark a scope component input as [resource](https://mscosmos.visualstudio.com/CosmosWiki/_wiki/wikis/Cosmos.wiki/565/RESOURCE). Refer [scope dynamic resources](./scope_component.html#dynamic-resources) for details.|
|datastore_mode| String                  | No       | The mode that will be used for this input. For File Dataset, available options are 'mount', 'download' and 'direct', for Tabular Dataset, available options is 'direct'. See https://aka.ms/dataset-mount-vs-download for more details.|

### Parameter

Defines a parameter of the component. Refer to [here](component-spec-topics/inputs-and-outputs.md) for details.

| Name         | Type    | Required | Description                                                  |
| ------------ | ------- | -------- | ------------------------------------------------------------ |
| type         | String  | Yes      | Defines the type of this data. Refer to [Data Types for Parameters](#data-types-for-parameters) for details. |
| optional     | Boolean | No       | Indicates whether this input is optional. Default value is `False`. |
| default      | Dynamic | No       | The default value for this parameter. The type of this value is dynamic. e.g. If `type` field in Input is `Integer`, this value should be `Inteter`. If `type` is `String`, this value should also be `String`. This field is optional, defaults to `null` or `None` if not specified. |
| description  | String  | No       | Detailed description to the parameter.                       |
| min          | Numeric | No       | The minimum value that can be accepted. This field only takes effect when `type` is `Integer` or `Float`. Specify `Integer` or `Float` values accordingly. |
| max          | Numeric | No       | The maximum value that can be accepted. Similar to `min`.    |
| enum  | List    | No       | The acceptable values for the parameter. This field only takes effect when `type` is `Enum`. |


### Output

Defines an output port of the component. Refer to [here](component-spec-topics/inputs-and-outputs.md) for details.

| Name         | Type   | Required | Description                                                  |
| ------------ | ------ | -------- | ------------------------------------------------------------ |
| type         | String | Yes      | Defines the data type(s) of this output port. Refer to [Data Types for Inputs/Outputs](#data-types-for-inputs/outputs) for details. |
| description  | String | No       | Detailed description to the output port.                     |
| is_link_mode | Boolean| No       | Set to true to mark an output to link an existing dataset as the output of current component, in runtime only "link" mode can be used. Refer to [Example Usage for Inputs/Outputs](https://github.com/Azure/DesignerPrivatePreviewFeatures/blob/master/azure-ml-components/samples/components/link-output-dataset/component.yaml) for details. |
|datastore_mode| String | No       | Specify whether to use 'upload', 'mount' or 'link' to access the data. Note that 'mount' and 'link' only works in a linux compute, windows compute only supports 'upload'. If 'upload' is specified, the output data will be uploaded to the datastore after the component process ends; If 'mount' is specified, all the updates of the output folder will be synced to the datastore when the component process is writting the output folder. If 'link' is specified, it will link an existed dataset as the output of current component.|

### Command

Command is a string that specify the command line to run the component.
It is expected to be a ***one-line string*** in which the arguments are separated by spaces.
The string will be split to a command list according to the shell split rule with the python built-in function [shlex.split](https://docs.python.org/3/library/shlex.html).

Example:

```yaml
command: >-
  python basic_component.py
  --input_dir {inputs.input_dir}
  --str_param {inputs.str_param}
  --enum_param {inputs.enum_param}
  --output-eval-dir {outputs.output_dir}
```

#### Yaml String Format
In the yaml file, it is recommended to use the ***folded*** style with the indicator `>-` to write a one-line string as multiple lines.

If the ***literal*** style with the indicator `|` is used, the command will contain `\n`, which could be handled, but is not recommended.

Unlike programming languages, yaml doesn't use `\` to indicate an unfinished line but treats it as a normal character. A `\` at the end of one line is not recommended since it could not be recognized as an unfinished line in a yaml string.

See [reference](https://yaml-multiline.info/) for more details of the yaml multi-line format.

#### CLI Argument Place Holders

When invoking from a CLI interface, the arguments are specified with placeholders like `{inputs.input_dir}`. The placeholders will be replaced with the actual value when running.

For example, when we set `input_dir='./input'`, the command `--input_dir {inputs.input_dir}` will be replaced as `--input_dir ./input`.

Placeholders are with this format: `inputs.input_name`/`outputs.output_name`.

As for optional inputs, the placeholders should be like `[--optional-input-path {inputs.optional_input_path}]` or `[--optional-input-path={inputs.optional_input_path}]`.
See [reference](./component-spec-topics/inputs-and-outputs.html#optional-inputs-and-parameters) for more details.

The following table lists some scenarios supported by argument place holder:

| Scenario                                                     | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| python train.py {inputs.inputFile}                           | Simple scenario to run a command with argument passed as parameter or input dataset. |
| python train.py {inputs.DataFolder}/data.csv                 | Scenario to run a command with data folder argument passed as parameter or input dataset and interpolation with the file name. |
| python train.py {inputs.DataFolder}/{inputs.inputFile}       | Scenario to run a command with data folder argument passed as input dataset and interpolation with the file name passed as parameter. |
| python train.py "{inputs.inputFile}"                         | Scenario to run a command with parameter or input dataset for file containing spaces or (, ), {, }. |
| python train.py foo.bar={inputs.input1}                      | Scenario to run a command with no space supported argument passed as parameter or input dataset. |
| Cool.exe [--param1 {inputs.param1}] [--param2={inputs.param2}] | Scenario to run a command with optional parameters.          |

#### Notice

* The command should follow the command line constraint on the corresponding OS, in a linux compute, it should follow [Shell Command Language](https://pubs.opengroup.org/onlinepubs/009604499/utilities/xcu_chap02.html), in a windows compute, it should follow [Command-Line Reference](https://docs.microsoft.com/en-us/previous-versions/windows/it-pro/windows-server-2012-r2-and-2012/cc754340(v=ws.11));
* Even the command doesn't use python, the image or the conda must contain the dependency "azureml-defaults" to run the command;
* If in some scenarios, the python style command raise error by specific characters, you can set component type as `CommandComponent@1-legacy` to execute component, please see [reference](./command_component.html#how-to-write-commandcomponent-yaml-spec) for more details of `CommandComponent@1-legacy`;
* Special characters escaping:
  - The special placeholder characters that are `[` `]` and `{` `}`
  - The escaping for these characters is by doubling of the specific character: `[[`  `]]` and `{{`  `}}`.


### Successful return code
Successful return code is used to specify how command return code is interpreted when component type is not [CommandComponent@1-legacy](./command_component.html#how-to-write-commandcomponent-yaml-spec). A non successful return code means the run will fail due to user error.

It only supports "Zero" and "ZeroOrGreater", default to "Zero" if not specified. And "ZeroOrGreater" is used to be compatible with some [Historically modules](https://aetherwiki.azurewebsites.net/articles/FeatureAreas/Non-Zero_Return_Codes.html).

If "Zero", zero return code means success, any other value is considered a user error.

If "ZeroOrGreater", zero or greater return code means success, a negative value is considered a user error.

### Environment

An Environment defines the runtime environment for the component to run, it is equivalent with the definition of the [Environment class in python SDK](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.environment%28class%29?view=azure-ml-py).

| Name      | Type                    | Required | Description                                                  |
| --------- | ----------------------- | -------- | ------------------------------------------------------------ |
| docker | DockerSection | No       | This section configures settings related to the final Docker image built to the specifications of the environment and whether to use Docker containers to build the environment. |
| conda | CondaSection | No       | This section specifies which Python environment and interpreter to use on the target compute. |           |
| os        | String                  | No       | Defines the operating system the component running on. Could be `windows` or `linux`. Defaults to `linux` if not specified. |


#### DockerSection

| Name      | Type   | Required | Description                                                  |
| --------- | ------ | -------- | ------------------------------------------------------------ |
| image | String | No       | The base image used for Docker-based runs. Example value: "ubuntu:latest". If not specified, will use `mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04` by default. |

#### CondaSection

| Name                  | Type              | Required | Description                                                  |
| --------------------- | ----------------- | -------- | ------------------------------------------------------------ |
| conda_dependencies_file | String            | No       | The path to the conda dependencies file to use for this run. If a project contains multiple programs with different sets of dependencies, it may be convenient to manage those environments with separate files. The default is None. |
| conda_dependencies     | CondaDependencies | No       | Same as `conda_dependencies_file`, but it is specifies the conda dependencies using an inline dictionary rather than a separated file. |
| pip_requirements_file | String | No | The path to the pip requirements file. |

### HDInsight

This section is used only for HDInsight components.

| Name            | Type         | Required | Description                                                  |
| --------------- | ------------ | -------- | ------------------------------------------------------------ |
| file            | String       | Yes      | File containing the application to execute, can be a python script or a jar file. It's the entry file of component. Specify a relative path to the code folder. |
| files           | List<String> | No       | Files to be placed in the working directory of each HDI executor. Support local files (relative paths to the code folder), HDFS compatible file system URIs (like `wasbs://file`) and public URIs (like https://file). |
| class_name       | String       | No       | Main class name when main file is a jar.                     |
| jars            | List<String> | No       | Jar files to be included on the HDI driver and executor classpaths. Support local files (relative paths to the code folder), HDFS compatible file system URIs (like `wasbs://file`) and public URIs (like https://file). |
| py_files         | List<String> | No       | List of .zip, .egg, or .py files to be placed on the PYTHONPATH for Python apps. Support local files (relative paths to the code folder), HDFS compatible file system URIs (like `wasbs://file`) and public URIs (like https://file). |
| archives        | List<String> | No       | Archives to be extracted into the working directory of each HDI executor. Support local files (relative paths to the code folder), HDFS compatible file system URIs (like `wasbs://file`) and public URIs (like https://file). |
| args            | String         | No       | Specify the arguments used along with `file`. This list may consist place holders of Inputs and Outputs. See [CLI Argument Place Holders](#cli-argument-place-holders) for details. |


And the followings can be overridden by [HDInsight RunSettings](../concepts/runsettings.html#hdinsight).

| Name            | Type         | Required | Description                                                  |
| --------------- | ------------ | -------- | ------------------------------------------------------------ |
| queue           | String       | No       | The name of the YARN queue to which submitted. |
| driver_memory   | String       | No       | Amount of memory to use for the driver process. It's the same format as JVM memory strings. Use lower-case suffixes, e.g. k, m, g, t, and p, for kibi-, mebi-, gibi-, tebi-, and pebibytes, respectively. Example values are 10k, 10m and 10g. |
| driver_cores    | Int          | No       | Number of cores to use for the driver process. |
| executor_memory | String       | No       | Amount of memory to use per executor process. It's the same format as JVM memory strings. Use lower-case suffixes, e.g. k, m, g, t, and p, for kibi-, mebi-, gibi-, tebi-, and pebibytes, respectively. |
| executor_cores  | Int          | No       | Number of cores to use for each executor. |
| number_executors | Int         | No       | Number of executors to launch for this session. |
| conf            | Dictionary<String, String> | No | Spark configuration properties. |
| name            | String       | No       | The name of this session. |

> **Note**
>
> HDInsight components are only for internal use currently.
>
> HDInsight components only work on internal compliant HDI cluster created by Office team for now.

### Scope

This section is used only for scope components.

| Name            | Type         | Required | Description                                                  |
| --------------- | ------------ | -------- | ------------------------------------------------------------ |
|     script           | String  |   Yes    | Specify the scope script to be executed. |
|     args             | String  |   Yes    | Specify the argument name of component's inputs and outputs. |
|  adla_account_name   | String  |   No     | Specify the default ADLA account name to use for the scope job.|
|  scope_param         | String  |   No     | Specify the default nebula command used when submit the scope job.|
|custom_job_name_suffix| String  |   No     | Specify the default string to append to scope job name.|

> **Note**
>
> Scope components are only for internal use currently.
>
> Prefix or postfix are not supported when defining scope args, e.g. `input_argname {inputs.input1}.tsv`
>
> Please refere to [scope yaml sample](./scope_component.html#how-to-write-scopecomponent-yaml-spec) for how to write scope args section.

### Starlite

This section is used only for starlite components.

| Name            | Type         | Required | Description                                                  |
| --------------- | ------------ | -------- | ------------------------------------------------------------ |
|command| String| Yes| Specify the command line to be executed. |
|starlite| Dictionary<String, String>| Yes| Must contain ref_id: `Your-AEther-Starlite-Module-Id`. |

> **Note**
>
> Starlite components are for internal use only.
>
> The component must reference an existing Starlite module. The constraint is because the Starlite cluster relies on AEther module registration, and the maintenance team wants to reserve the capability of creating/modifying modules to their own.
>
> The inputs and outputs must be data in Azure Data Lake on Cosmos under "local" folder.
>
> Please refer to [starlite yaml sample](./starlite_component.html#how-to-write-starlitecomponent-yaml-spec) for how to write starlite args section.

### AE365ExePool

This section is used only for AE365ExePool components.

| Name            | Type         | Required | Description                                                  |
| --------------- | ------------ | -------- | ------------------------------------------------------------ |
|ae365exepool| Dictionary<String, String>| Yes| Must contain ref_id: `Your-AEther-AE365ExePool-Module-Id`. |

> **Note**
>
> AE365ExePool components are for internal use only.
>
> The component must reference an existing AEther AE365ExePool module. Currently only CAX EyesOn Module [ND] v1.6 (654ec0ba-bed3-48eb-a594-efd0e9275e0d) is supported.
>
> The inputs and outputs must be data in Azure Data Lake on Cosmos under "local" folder.
>
> Please refer to [ae365exepool yaml sample](./ae365exepool_component.html#how-to-write-ae365exepoolcomponent-yaml-spec) for how to write args section.

### AetherBridge

This section is used only for AetherBridge components.

| Name            | Type         | Required | Description                                                  |
| --------------- | ------------ | -------- | ------------------------------------------------------------ |
|command| String| Yes| Specify the command line to be executed. |
|aether| Dictionary<String, String>| Yes| Must contain module_type: `Your-AEther-Module-Type`, ref_id: `Your-AEther-Module-Id`. |

> **Note**
>
> AetherBridge components are for internal use only.
>
> The component must reference an existing Aether module.
>
> AetherBridge component is a temporary approach, we will provide AML native solution in long-term. If your Aether module needs to go with AEtherBridge component, please follow [this form](https://forms.office.com/Pages/ResponsePage.aspx?id=v4j5cvGGr0GRqy180BHbRyuCORqKEQ5FiTsyzbdvFThUMllJRTdDVTFLTlRPQU0zQUdSNlVJRFlVUiQlQCN0PWcu) to register your ask.
>
> The inputs and outputs must be data in Azure Data Lake on Cosmos under "local" folder.
>
> Please refer to [aetherbridge yaml sample](./aetherbridge_component_for_scrpaing_job.html#How_to_write_AetherBridge_component_yaml_spec) for how to write aetherbridge args section.

### Parallel

This section is used only for parallel components.
Parallel component is a kind of component to run [ParallelRunStep](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-steps/azureml.pipeline.steps.parallelrunstep?view=azure-ml-py#constructor).

| Name        | Type                   | Required | Description                                                  |
| ----------- | ---------------------- | -------- | ------------------------------------------------------------ |
| input_data  | String or List<String> | Yes      | The input(s) provide the data to be split into mini_batches for parallel execution. Specify the name(s) of the corresponding input(s) here, note that the input(s) not in input_data are 'side_input' in ParallelRunStep concept. |
| output_data | String                 | Yes      | The output for the summarized result that generated by the user script. Specify the name of the corresponding output here. |
| entry       | String                 | Yes      | The user script to process mini_batches.                     |
| args        | String                 | No       | Specify the arguments used along with `entry`. This list may consist place holders of Inputs and Outputs. See [CLI Argument Place Holders](#cli-argument-place-holders) for details. |

### Hemera

This section is used only for Hemera components.

| Name   | Type   | Required | Description                                                  |
| ------ | ------ | -------- | ------------------------------------------------------------ |
| ref_id | String | Yes      | Reference to an existed Aether module.<br />ref_id: `Your-AEther-Hemera-Module-Guid`. |

> **Note**
>
> Hemera components are for internal use only.
>
> The component must reference an existing Hemera module. It's because the Hemera backend cluster relies on AEther module registration.
>
> The input and output paths must be native Cosmos paths that can be visible on Cosmos portal, e.g. "/local" folder.
>
> Please refer to [hemera yaml sample](./hemera_component.html#how-to-write-hemeracomponent-yaml-spec).

### launcher

This section is used only for DistributedComponents.
DistributedComponent is a kind of component to support distributed training scenarios.

| Name        | Type                   | Required | Description                                                  |
| ----------- | ---------------------- | -------- | ------------------------------------------------------------ |
| type  | String | Yes      | Launch type of a distributed training, Could be `mpi`, `torch.distributed`. |
| additional_arguments | String   | Yes      | The command to invoke custom script. |

### Data Types

Data Type is a short word or phrase that describes the data type of the Input or Output.

#### Data Types for Inputs/Outputs

Designer allows its user to connect an Output to another component's Input with the same data type.

The data type for an Input/Output could be an arbitrary string (except `<` and `>`).

Below is a list of data types that will be auto-registered and can be directly used by users out-of-box.
For other data type names, please create the DataTypes first following guide from https://aka.ms/azureml-sdk-create-data-type.

**Data Types**

| Name                   | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| path                   | A path contains arbitray data.                               |
| AzureMLDataset         | Represents a dataset, passed directly as id in command line. |

**Data Types used by built-in components**

| Name                   | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| AnyDirectory           | Generic directory which stores arbitray data                 |
| DataFrameDirectory     | Represents tabular data, saved in parquet format by default. |
| ModelDirectory         | Represents a trained model. can be in any format or flavor, will have its own spec file to describe the detailed information. |
| ImageDirectory         | Store images and related meta data in the directory.         |
| UntrainedModelDirectory| Represents an untrained model.                                |
| TransformationDirectory| Represents a transform, only for backward compatibility.     |
| AnyFile                | Generic text/binary file.                                    |
| ZipFile                | A Zipped File.                                               |
| CsvFile                | A CSV or TSV format, with or without header, zipped (of a single file) or unzipped. |

**Data Types used by scope components**

| Name                   | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| CosmosStructuredStream | Represents cosmos structured stream.                         |

#### Data Types for Parameters

| Name    | Description                                                  |
| ------- | ------------------------------------------------------------ |
| String  | Indicates that the input value is a string.                  |
| Integer | Indicates that the input value is a 64-bit signed integer, values out of range will cause an exception.                |
| Float   | Indicates that the input value is a 64-bit signed floating-point number, floating-numbers more than 64-bit will lose accuracy, non-number values will cause an exception. |
| Boolean | Indicates that the input value is a boolean value, it should be 'True' or 'False', all other values will cause an exception. |
| Enum    | Indicates that the input value is a enumerated (limited list of) String values. |

### Spark

This section is used only for Spark components.

| Name            | Type         | Required | Description                                                  |
| --------------- | ------------ | -------- | ------------------------------------------------------------ |
| entry:<br/> file:           | String       | Yes      | File containing the application to execute, can be a python script. It's the entry file of component. Specify a relative path to the code folder. |
| files           | List<String> | No       | Files to be placed in the working directory of each spark executor. Specify local files (relative paths to the code folder) |
| jars            | List<String> | No       | Jar files to be included on the spark driver and executor classpaths. Specify local files (relative paths to the code folder) |
| py_files         | List<String> | No       | List of .zip, .egg, or .py files to be placed on the PYTHONPATH for Python apps. Specify local files (relative paths to the code folder) |
| archives        | List<String> | No       | Archives to be extracted into the working directory of each spark executor. Specify local files (relative paths to the code folder) |
| args            | String         | No       | Specify the arguments used along with `file`. This list may consist place holders of Inputs and Outputs. See [CLI Argument Place Holders](#cli-argument-place-holders) for details. |
| conda_dependencies            | CondaDependencies         | No       | Specify the inline conda dependencies you need for this spark job |


| Name            | Type         | Required | Description                                                  |
| --------------- | ------------ | -------- | ------------------------------------------------------------ |
| identity| IdentitySetting       | No       | Specify which identity is used to run the spark job.|
| driver_memory   | String       | No       | Amount of memory to use for the driver process. It's the same format as JVM memory strings. Use lower-case suffixes, e.g. k, m, g, t, and p, for kibi-, mebi-, gibi-, tebi-, and pebibytes, respectively. Example values are 10k, 10m and 10g. |
| driver_cores    | Int          | No       | Number of cores to use for the driver process. |
| executor_memory | String       | No       | Amount of memory to use per executor process. It's the same format as JVM memory strings. Use lower-case suffixes, e.g. k, m, g, t, and p, for kibi-, mebi-, gibi-, tebi-, and pebibytes, respectively. |
| executor_cores  | Int          | No       | Number of cores to use for each executor. |
| number_executors | Int         | No       | Number of executors to launch for this session. |
| conf            | Dictionary<String, String> | No | Spark configuration properties. |

#### IdentitySetting

| Name      | Type   | Required | Description                                                  |
| --------- | ------ | -------- | ------------------------------------------------------------ |
| Type | Enum | No       | By default it's user identity, you can choose user_identity or managed in the type property.|

> **Note**
>
> Spark components (1.5) are only for internal use currently, it will be available in dpv2 in Sep/2022. Till then it will be available for 3p customers.
>

