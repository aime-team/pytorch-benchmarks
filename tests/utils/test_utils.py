from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from utils.utils import Protocol, dt_now_to_str


@pytest.fixture
def protocol_fixture():
    rank = "test rank"
    args = Mock()
    args.device = ""
    args.load_from_epoch = 0
    args.no_benchmark = False

    model = "test model"
    data = "test data"

    return Protocol(rank, args, model, data)


@pytest.mark.parametrize(
    "input_time,expected",
    [
        ((2020, 3, 13, 10, 22, 7), "2020-03-13 10:22:07.00"),
        ((2025, 2, 19, 16, 55, 3, 708629), "2025-02-19 16:55:03.70"),
        ((2025, 2, 19, 16, 55, 3, 700000), "2025-02-19 16:55:03.70"),
        ((2030, 1, 1), "2030-01-01 00:00:00.00")
    ]
)
def test_dt_now_to_str(input_time, expected):
    """Tests dt_now_to_str with different input times"""

    # Arrange
    mocked_datetime = Mock()
    mocked_datetime.now.configure_mock(return_value=datetime(*input_time))

    # Act
    with patch("utils.utils.datetime", mocked_datetime):
        actual = dt_now_to_str()

    # Assert
    assert actual == expected


class TestProtocolMakeProgressPromptString:
    """
    Tests Protocol's class make_progress_prompt_string method.
    """
    def test_base(self, protocol_fixture):
        """Basic Test"""
        # Arrange
        mocked_dt_now_str = Mock()
        mocked_dt_now_str.configure_mock(return_value="dummy_dt_string")

        epoch = 1
        step = 2
        total_steps = 3500
        it_per_sec = 325.7
        protocol_fixture.eval_mode = True
        protocol_fixture.args.iterations = "Dummy_Images"
        protocol_fixture.args.num_epochs = 5

        # Act
        with patch("utils.utils.dt_now_to_str", mocked_dt_now_str):
            result = protocol_fixture.make_progress_prompt_string(
                epoch, step, total_steps, it_per_sec=it_per_sec
            )

        # Assert
        expected = "dummy_dt_string Epoch 1/5, Step 2/3500, 325.7 Dummy_Images/sec\n"
        assert result == expected

    def test_with_loss(self, protocol_fixture):
        """Test where loss is specified"""
        # Arrange
        mocked_dt_now_str = Mock()
        mocked_dt_now_str.configure_mock(return_value="dummy_dt_string")

        epoch = 1
        step = 2
        total_steps = 3500
        loss_value = 6.834556
        loss = Mock()
        detach = Mock()
        detach.item.configure_mock(return_value=loss_value)
        loss.detach.configure_mock(return_value=detach)
        it_per_sec = 325.7
        protocol_fixture.eval_mode = False
        protocol_fixture.args.iterations = "Dummy_Images"
        protocol_fixture.args.num_epochs = 5

        # Act
        with patch("utils.utils.dt_now_to_str", mocked_dt_now_str):
            result = protocol_fixture.make_progress_prompt_string(
                epoch, step, total_steps, loss=loss, it_per_sec=it_per_sec
            )

        # Assert
        expected = "dummy_dt_string Epoch 1/5, Step 2/3500, Loss: 6.8346, 325.7 Dummy_Images/sec\n"
        assert result == expected
        loss.detach.assert_called_once_with()
        detach.item.assert_called_once_with()


class TestProtocolMakeInfoText:
    """
    Tests Protocol's class make_info_text method.
    """
    def test_base(self, protocol_fixture):
        """Basic test"""
        # Arrange
        mocked_check_output = Mock()
        model_mocked_value = (
            "ignored\n"
            "Model name:       Faketel(R) Core(TM) i5-6500 CPU @ 3.20GHz\n"
            "ignored\n"
        )
        mocked_check_output.configure_mock(return_value=model_mocked_value.encode())
        mocked_dt_now_str = Mock()
        mocked_dt_now_str.configure_mock(return_value="dummy_dt_string")

        mocked_platform = Mock()
        mocked_uname = Mock()
        mocked_uname.system = "Dummy_OS"
        mocked_uname.release = "Dummy_release"
        mocked_uname.node = "Dummy_device_name"
        mocked_platform.uname.configure_mock(return_value=mocked_uname)

        protocol_fixture.rank = 0
        protocol_fixture.args.model = "Dummy_Model"
        protocol_fixture.args.global_batch_size = 8
        protocol_fixture.args.batch_size = 16
        protocol_fixture.args.global_eval_batch_size = 32
        protocol_fixture.args.eval_batch_size = 63
        protocol_fixture.args.distribution_mode = 0
        protocol_fixture.args.process_group_backend = "dummy_ncc1"
        protocol_fixture.args.auto_mixed_precision = True
        protocol_fixture.args.compile = True
        protocol_fixture.args.log_file = "dummy_1_NVIDIAGeForceGTX1650_resnet50_64_lr0001.txt"
        protocol_fixture.args.data_name = "dummy Synthetic data"
        protocol_fixture.args.learning_rate = 0.00172
        protocol_fixture.args.step_lr = 29
        protocol_fixture.args.no_augmentation = False
        protocol_fixture.args.checkpoint_folder = (
            "/pytorch-benchmarks/model_checkpoints/"
            "dummy_1_NVIDIAGeForceGTX1650_resnet50_64_lr0001"
        )
        protocol_fixture.args.num_workers = 7
        protocol_fixture.args.warm_up_steps = 11

        # Act
        with (
            patch("utils.utils.subprocess.check_output", mocked_check_output),
            patch("utils.utils.dt_now_to_str", mocked_dt_now_str),
            patch("utils.utils.platform", mocked_platform)
        ):
            result = protocol_fixture.make_info_text()

        # Assert
        expected = """OS: Dummy_OS, Dummy_release
Device-name: Dummy_device_name
CPU used for benchmark: Faketel(R) Core(TM) i5-6500 CPU @ 3.20GHz
Available GPUs on device: 1
Cuda-version: 12.4
Cudnn-version: 90100
Python-version: 3.12.3
PyTorch-version: 2.6.0+cu124
CPU: Faketel(R) Core(TM) i5-6500 CPU @ 3.20GHz
Model: Dummy_Model
Global train batch size: 8
Local train batch size: 16
Global evaluation batch size: 32
Local evaluation batch size: 63
Distribution Mode: Single GPU Training 
Process group backend: dummy_ncc1
Optimizer: ZeroRedundancyOptimizer with SGD
Precision: Automatic mixed precision
Compile-Mode: True
Log file: dummy_1_NVIDIAGeForceGTX1650_resnet50_64_lr0001.txt
Training data: dummy Synthetic data
Initial learning rate: 0.00172
Learning rate decay step: 29
Used data augmentation: True
Checkpoint folder: /pytorch-benchmarks/model_checkpoints/dummy_1_NVIDIAGeForceGTX1650_resnet50_64_lr0001
Number of workers: 7
Warm up steps: 11
Benchmark start : dummy_dt_string

"""
        assert result == expected


class TestProtocolFinishString:
    """
    Tests Protocol's class finish_string method.
    """
    def test_base(self, protocol_fixture):
        """Basic test"""
        # Arrange
        mocked_dt_now_str = Mock()
        mocked_dt_now_str.configure_mock(return_value="dummy_dt_string")

        # Act
        with patch("utils.utils.dt_now_to_str", mocked_dt_now_str):
            result = protocol_fixture.finish_string()

        # Assert
        expected = "\n\n\nBenchmark end: dummy_dt_string\n"
        assert result == expected

    def test_with_mean_it_per_sec(self, protocol_fixture):
        """Test when mean iteration per second argument is set"""
        # Arrange
        protocol_fixture.args.mean_it_per_sec = True
        protocol_fixture.benchmark = Mock()
        protocol_fixture.benchmark.make_final_mean_it_per_sec_string.configure_mock(return_value="dummy mean_it_per_sec string")
        protocol_fixture.gpu_info = False

        mocked_dt_now_str = Mock()
        mocked_dt_now_str.configure_mock(return_value="dummy_dt_string")

        # Act
        with patch("utils.utils.dt_now_to_str", mocked_dt_now_str):
            result = protocol_fixture.finish_string()

        # Assert
        expected = "dummy mean_it_per_sec string\n\nBenchmark end: dummy_dt_string\n"
        assert result == expected

    def test_with_gpu_info(self, protocol_fixture):
        """Test when gpu_info is present"""
        # Arrange
        protocol_fixture.args.mean_it_per_sec = False
        protocol_fixture.gpu_info = Mock()
        protocol_fixture.gpu_info.get_max_temperature_str.configure_mock(return_value="dummy max_temperature string")

        mocked_dt_now_str = Mock()
        mocked_dt_now_str.configure_mock(return_value="dummy_dt_string")

        # Act
        with patch("utils.utils.dt_now_to_str", mocked_dt_now_str):
            result = protocol_fixture.finish_string()

        # Assert
        expected = "dummy max_temperature string\n\nBenchmark end: dummy_dt_string\n"
        assert result == expected

    def test_with_all(self, protocol_fixture):
        """Test when all the options that impact this string are specified"""
        # Arrange
        protocol_fixture.args.mean_it_per_sec = True
        protocol_fixture.benchmark = Mock()
        protocol_fixture.benchmark.make_final_mean_it_per_sec_string.configure_mock(return_value="dummy mean_it_per_sec string")
        protocol_fixture.gpu_info = Mock()
        protocol_fixture.gpu_info.get_max_temperature_str.configure_mock(return_value="dummy max_temperature string")
        mocked_dt_now_str = Mock()
        mocked_dt_now_str.configure_mock(return_value="dummy_dt_string")

        # Act
        with patch("utils.utils.dt_now_to_str", mocked_dt_now_str):
            result = protocol_fixture.finish_string()

        # Assert
        expected = "dummy mean_it_per_sec stringdummy max_temperature string\n\nBenchmark end: dummy_dt_string\n"
        assert result == expected
