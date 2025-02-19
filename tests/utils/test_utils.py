from unittest.mock import Mock

import pytest
from utils.utils import Protocol


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


class TestProtocolMakeProgressPromptString:
    """
    Tests Protocol's class make_progress_prompt_string method.
    """
    def test_base(self, protocol_fixture):
        """Basic Test"""
        # Arrange
        epoch = 1
        step = 2
        total_steps = 3500
        it_per_sec = 325.7
        protocol_fixture.eval_mode = True
        protocol_fixture.args.iterations = "Dummy_Images"
        protocol_fixture.args.num_epochs = 5

        # Act
        result = protocol_fixture.make_progress_prompt_string(
            epoch, step, total_steps, it_per_sec=it_per_sec
        )

        # Assert
        expected = "Epoch [1 / 5], Step [2 / 3500],  Dummy_Images per second: 325.7\n"
        assert result == expected

    def test_with_loss(self, protocol_fixture):
        """Test where loss is specified"""
        # Arrange
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
        result = protocol_fixture.make_progress_prompt_string(
            epoch, step, total_steps, loss=loss, it_per_sec=it_per_sec
        )

        # Assert
        expected = "Epoch [1 / 5], Step [2 / 3500], Loss: 6.8346,  Dummy_Images per second: 325.7\n"
        assert result == expected
        loss.detach.assert_called_once_with()
        detach.item.assert_called_once_with()
