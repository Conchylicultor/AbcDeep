# AbcDeep
Module to simplify the creation of Deep Learning research programs with TensorFlow.

To use the module, those *Abstract Base Classes* needs to be overloaded:
 * AbcProgram: The main class which contains the main training loop
 * AbcModel: The class containing the network definition
 * AbcDataManager: Dataset creation and loading

The module require the following dependencies:
 * tensorflow 1.0
 * tqdm
