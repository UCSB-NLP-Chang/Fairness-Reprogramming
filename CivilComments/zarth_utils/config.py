import os
import json
import argparse
import logging

from zarth_utils.general_utils import get_random_time_stamp, makedir_if_not_exist

dir_configs = os.path.join(os.getcwd(), "configs")


class Config:
    def __init__(self,
                 default_config_file=os.path.join(os.getcwd(), "default_config.json"),
                 default_config_dict=None,
                 use_argparse=True):
        """
        Initialize the config. Note that either default_config_dict or default_config_file in json format must be
        provided! If both are provided, only the dict will be used. The keys will be transferred to
        argument names, and the type will be automatically detected.
        The priority is the user specified parameter > user specified config file > default config file

        Examples:
        default_config_dict = {"lr": 0.01, "optimizer": "sgd", "num_epoch": 30, "use_early_stop": False}
        Then the following corresponding arguments will be added in this function:
        parser.add_argument("--lr", type=float)
        parser.add_argument("--optimizer", type=str)
        parser.add_argument("--num_epoch", type=int)
        parser.add_argument("--use_early_stop", action="store_true", default=False)
        parser.add_argument("--no-use_early_stop", dest="use_early_stop", action="store_false")

        :param default_config_dict: the default config dict
        :type default_config_dict: dict
        :param default_config_file: the default config file path
        :type default_config_file: str
        """
        self.__parameters = {}

        # load from default config file
        if default_config_dict is not None:
            self.__parameters.update(default_config_dict)
        else:
            self.__parameters.update(json.load(open(default_config_file, "r", encoding="utf-8")))

        if use_argparse:
            # add argument parser
            parser = argparse.ArgumentParser()
            parser.add_argument("--config_file", type=str, default=None)
            for name_param in self.__parameters.keys():
                value_param = self.__parameters[name_param]
                if type(value_param) is bool:
                    parser.add_argument("--%s" % name_param, action="store_true", default=value_param)
                    parser.add_argument("--no-%s" % name_param, dest="%s" % name_param, action="store_false")
                elif type(value_param) is list:
                    parser.add_argument("--%s" % name_param, type=type(value_param[0]), default=value_param, nargs="+")
                else:
                    parser.add_argument("--%s" % name_param, type=type(value_param), default=value_param)
            args = parser.parse_args()

            updated_parameters = dict()
            args_dict = vars(args)
            for k in vars(args):
                if k != "config_file" and self.__parameters[k] != args_dict[k]:
                    updated_parameters[k] = args_dict[k]

            if args.config_file is not None:
                self.__parameters.update(json.load(open(args.config_file, "r", encoding="utf-8")))

            self.__parameters.update(updated_parameters)

    def __getitem__(self, item):
        """
        Return the config[item]
        :param item: the key of interest
        :type item: str
        :return: config[item]
        """
        return self.__parameters[item]

    def to_dict(self):
        """
        Return the config as a dict
        :return: config dict
        :rtype: dict
        """
        return self.__parameters

    def show(self):
        """
        Show all the configs in logging. If get_logger is used before, then the outputs will also be in the log file.
        """
        logging.info("\n%s" % json.dumps(self.__parameters, sort_keys=True, indent=4, separators=(',', ': ')))

    def dump(self, path_dump=None):
        """
        Dump the config in the path_dump.
        :param path_dump: the path to dump the config
        :type path_dump: str
        """
        if path_dump is None:
            makedir_if_not_exist(dir_configs)
            path_dump = os.path.join(dir_configs, "%s.config" % get_random_time_stamp())
        path_dump = "%s.config" % path_dump if not path_dump.endswith(".config") else path_dump
        assert not os.path.exists(path_dump)
        json.dump(self.__parameters, open(path_dump, "w", encoding="utf-8"))

    def keys(self):
        return self.__parameters.keys()
