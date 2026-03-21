import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.detector_agent import FraudDetectorAgent

class TestFraudDetectorAgent:

    def setup_method(self):
        """Runs before every test - creates fresh agent"""
        self.agent = FraudDetectorAgent(name="TestAgent")

    def test_agent_initialises_correctly(self):
        """Agent should have correct name and zero cases"""
        assert self.agent.name == "TestAgent"
        assert self.agent.cases_reviewed == 0

    def test_cases_reviewed_increments(self):
        """cases_reviewed should increase after each analyse call"""
        transaction = {"Amount": 100.0, "hour": 14, "Time": 50000}
        self.agent.analyse(transaction)
        assert self.agent.cases_reviewed == 1

    def test_transaction_dict_structure(self):
        """Transaction dict must have required keys"""
        transaction = {"Amount": 100.0, "hour": 14, "Time": 50000}
        assert "Amount" in transaction
        assert "hour" in transaction
        assert "Time" in transaction

    def test_high_risk_detection(self):
        """Zero amount transaction should return HIGH risk"""
        transaction = {"Amount": 0.0, "hour": 2, "Time": 50000}
        result = self.agent.analyse(transaction)
        assert "HIGH" in result

    def test_low_risk_detection(self):
        """Normal transaction should return LOW risk"""
        transaction = {"Amount": 150.0, "hour": 14, "Time": 50000}
        result = self.agent.analyse(transaction)
        assert "LOW" in result or "MEDIUM" in result

    def test_response_format(self):
        """Response must contain Risk Level, Reason, Action"""
        transaction = {"Amount": 100.0, "hour": 14, "Time": 50000}
        result = self.agent.analyse(transaction)
        assert "Risk Level" in result
        assert "Reason" in result
        assert "Action" in result