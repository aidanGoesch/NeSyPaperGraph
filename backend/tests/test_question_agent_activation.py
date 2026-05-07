import asyncio

from services.graph_builder import create_dummy_graph
from services.question_agent import QuestionAgent


def test_answer_question_with_activation_runs_second_round_when_low_confidence(monkeypatch):
    graph = create_dummy_graph()
    agent = QuestionAgent(graph_obj=graph)

    monkeypatch.setattr(agent, "_embed_text", lambda _text: [1.0, 0.1, 0.1])
    monkeypatch.setattr(
        agent,
        "_plan_activation_query",
        lambda current_question, conversation_history=None: current_question,
    )

    def fake_generate_answer(state):
        state["answer"] = "Grounded answer draft"
        state["answer_structured"] = {
            "segments": [{"text": "Grounded answer draft", "claim_id": None}],
            "claims": [],
            "warnings": [],
        }
        return state

    monkeypatch.setattr(agent, "_generate_answer", fake_generate_answer)

    confidence_values = iter([0.2, 0.84])
    monkeypatch.setattr(
        agent,
        "_estimate_confidence",
        lambda _answer, _nodes: next(confidence_values),
    )

    result = asyncio.run(
        agent.answer_question_with_activation(
            "How does this topic connect?",
            conversation_history=[{"question": "Earlier", "answer": "Context"}],
            confidence_threshold=0.6,
            max_rounds=2,
            seed_count=3,
        )
    )
    assert result["final_answer"] == "Grounded answer draft"
    assert result["answer_structured"]["segments"][0]["text"] == "Grounded answer draft"
    assert len(result["rounds"]) == 2
    assert result["needs_more_context"] is False
    assert result["confidence"] == 0.84
