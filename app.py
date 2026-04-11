#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse
from uuid import uuid4


ENV_NAME = "invoice-exception-openenv"
DEFAULT_PORT = int(os.environ.get("PORT", "7860"))
MIN_STRICT_SCORE = 0.01
MAX_STRICT_SCORE = 0.99

AVAILABLE_ACTIONS = [
    "review_invoice",
    "compare_po",
    "check_receipt",
    "search_duplicate",
    "request_supporting_doc",
    "approve",
    "route_to_review",
    "flag_duplicate",
    "reject",
    "done",
]

EVIDENCE_LABELS = {
    "reviewed_invoice": "invoice review",
    "checked_po": "purchase order comparison",
    "checked_receipt": "goods receipt verification",
    "checked_duplicates": "duplicate search",
    "requested_doc": "supporting document review",
    "flagged_risk": "duplicate escalation",
}


@dataclass
class SupplierProfile:
    supplier_id: str
    supplier_name: str
    risk_level: str
    payment_terms: str
    notes: str


@dataclass
class PurchaseOrder:
    po_number: str
    sku: str
    quantity: int
    unit_price: float
    currency: str
    approved_change_note: Optional[str] = None


@dataclass
class GoodsReceipt:
    receipt_id: str
    quantity_received: int
    received_on: str
    warehouse_notes: str


@dataclass
class Invoice:
    invoice_id: str
    supplier_id: str
    amount: float
    currency: str
    quantity_billed: int
    unit_price: float
    due_date: str
    po_number: str
    duplicate_reference: Optional[str] = None
    header_note: str = ""


@dataclass
class TaskDefinition:
    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    supplier: SupplierProfile
    invoice: Invoice
    purchase_order: PurchaseOrder
    goods_receipt: GoodsReceipt
    duplicate_invoice_found: bool
    relevant_actions: List[str]
    required_evidence: List[str]
    nice_to_have_evidence: List[str]
    final_action: str
    risk_focus: str
    resolution_justification: str
    scoring_weights: Dict[str, float]


@dataclass
class Episode:
    session_id: str
    task: TaskDefinition
    step_count: int = 0
    done: bool = False
    total_reward: float = 0.0
    revealed_docs: Dict[str, bool] = field(
        default_factory=lambda: {
            "invoice": True,
            "purchase_order": False,
            "goods_receipt": False,
            "duplicate_search": False,
            "supporting_doc": False,
        }
    )
    milestones: Dict[str, bool] = field(
        default_factory=lambda: {
            "reviewed_invoice": False,
            "checked_po": False,
            "checked_receipt": False,
            "checked_duplicates": False,
            "requested_doc": False,
            "flagged_risk": False,
            "resolved_correctly": False,
            "closed_case": False,
        }
    )
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    last_action_result: Optional[str] = None
    last_action_error: Optional[str] = None
    final_decision: Optional[str] = None
    unsafe_resolution_attempts: int = 0
    irrelevant_actions: int = 0


TASKS: Dict[str, TaskDefinition] = {}
SESSIONS: Dict[str, Episode] = {}
SESSION_LOCK = Lock()
LAST_SESSION_ID: Optional[str] = None


def build_tasks() -> Dict[str, TaskDefinition]:
    return {
        "invoice_easy": TaskDefinition(
            task_id="invoice_easy",
            name="Standard Match Approval",
            difficulty="easy",
            description=(
                "A routine office-supplies invoice should be approved only after the "
                "agent confirms both the purchase order and the goods receipt."
            ),
            max_steps=6,
            supplier=SupplierProfile(
                supplier_id="SUP-100",
                supplier_name="Northwind Office Supply",
                risk_level="low",
                payment_terms="Net 30",
                notes="Long-standing supplier with low dispute rate.",
            ),
            invoice=Invoice(
                invoice_id="INV-2001",
                supplier_id="SUP-100",
                amount=2400.0,
                currency="USD",
                quantity_billed=120,
                unit_price=20.0,
                due_date="2026-04-30",
                po_number="PO-8100",
                header_note="Monthly restock for printer toner cartridges.",
            ),
            purchase_order=PurchaseOrder(
                po_number="PO-8100",
                sku="TONER-77",
                quantity=120,
                unit_price=20.0,
                currency="USD",
            ),
            goods_receipt=GoodsReceipt(
                receipt_id="GR-9001",
                quantity_received=120,
                received_on="2026-04-02",
                warehouse_notes="Received in full. No discrepancies recorded.",
            ),
            duplicate_invoice_found=False,
            relevant_actions=["review_invoice", "compare_po", "check_receipt"],
            required_evidence=["reviewed_invoice", "checked_po", "checked_receipt"],
            nice_to_have_evidence=[],
            final_action="approve",
            risk_focus="Validate the standard three-way match before approving payment.",
            resolution_justification="Invoice approved after a clean invoice, PO, and receipt match.",
            scoring_weights={
                "reviewed_invoice": 0.12,
                "checked_po": 0.2,
                "checked_receipt": 0.23,
                "resolved_correctly": 0.35,
                "closed_case": 0.09,
            },
        ),
        "invoice_medium": TaskDefinition(
            task_id="invoice_medium",
            name="Price Mismatch Escalation",
            difficulty="medium",
            description=(
                "A freight invoice is priced above the PO. The agent must verify the "
                "mismatch, confirm the receipt, request support, and route the case to review "
                "because no approved change order exists."
            ),
            max_steps=7,
            supplier=SupplierProfile(
                supplier_id="SUP-220",
                supplier_name="BlueRiver Freight",
                risk_level="medium",
                payment_terms="Net 15",
                notes="Occasional surcharges appear, but they require written approval.",
            ),
            invoice=Invoice(
                invoice_id="INV-3020",
                supplier_id="SUP-220",
                amount=4600.0,
                currency="USD",
                quantity_billed=10,
                unit_price=460.0,
                due_date="2026-04-18",
                po_number="PO-9920",
                header_note="Includes expedited lane surcharge according to supplier note.",
            ),
            purchase_order=PurchaseOrder(
                po_number="PO-9920",
                sku="FREIGHT-LANE-7",
                quantity=10,
                unit_price=400.0,
                currency="USD",
                approved_change_note=None,
            ),
            goods_receipt=GoodsReceipt(
                receipt_id="GR-4112",
                quantity_received=10,
                received_on="2026-04-04",
                warehouse_notes="Shipment received on time. No service exceptions logged.",
            ),
            duplicate_invoice_found=False,
            relevant_actions=["review_invoice", "compare_po", "check_receipt", "request_supporting_doc"],
            required_evidence=["reviewed_invoice", "checked_po", "checked_receipt", "requested_doc"],
            nice_to_have_evidence=[],
            final_action="route_to_review",
            risk_focus="Do not approve a higher price without documented authorization.",
            resolution_justification="Invoice routed to review because the surcharge lacks approved support.",
            scoring_weights={
                "reviewed_invoice": 0.1,
                "checked_po": 0.18,
                "checked_receipt": 0.15,
                "requested_doc": 0.2,
                "resolved_correctly": 0.27,
                "closed_case": 0.09,
            },
        ),
        "invoice_hard": TaskDefinition(
            task_id="invoice_hard",
            name="Duplicate Invoice Prevention",
            difficulty="hard",
            description=(
                "A high-risk supplier submitted an urgent replacement invoice that appears valid "
                "on the surface. The agent must confirm the PO, inspect the receipt history, run a "
                "duplicate search, flag the risk, and reject the duplicate."
            ),
            max_steps=8,
            supplier=SupplierProfile(
                supplier_id="SUP-330",
                supplier_name="Vertex Industrial Parts",
                risk_level="high",
                payment_terms="Net 45",
                notes="Recent duplicate billing incidents. Always search invoice history before payment.",
            ),
            invoice=Invoice(
                invoice_id="INV-7781",
                supplier_id="SUP-330",
                amount=9800.0,
                currency="USD",
                quantity_billed=14,
                unit_price=700.0,
                due_date="2026-04-25",
                po_number="PO-4418",
                duplicate_reference="INV-7710",
                header_note="Rush replacement components for line shutdown recovery.",
            ),
            purchase_order=PurchaseOrder(
                po_number="PO-4418",
                sku="MOTOR-AX9",
                quantity=14,
                unit_price=700.0,
                currency="USD",
            ),
            goods_receipt=GoodsReceipt(
                receipt_id="GR-2203",
                quantity_received=14,
                received_on="2026-03-28",
                warehouse_notes="Goods already received and reconciled against invoice INV-7710.",
            ),
            duplicate_invoice_found=True,
            relevant_actions=["review_invoice", "compare_po", "check_receipt", "search_duplicate", "flag_duplicate"],
            required_evidence=["reviewed_invoice", "checked_po", "checked_receipt", "checked_duplicates"],
            nice_to_have_evidence=["flagged_risk"],
            final_action="reject",
            risk_focus="Treat urgency as a distraction. Verify history before authorizing payment.",
            resolution_justification="Duplicate invoice rejected after receipt history and duplicate search confirmed prior payment.",
            scoring_weights={
                "reviewed_invoice": 0.08,
                "checked_po": 0.12,
                "checked_receipt": 0.16,
                "checked_duplicates": 0.22,
                "flagged_risk": 0.08,
                "resolved_correctly": 0.24,
                "closed_case": 0.09,
            },
        ),
        "invoice_expert": TaskDefinition(
            task_id="invoice_expert",
            name="Approved Change Order Clearance",
            difficulty="expert",
            description=(
                "A cold-chain shipment invoice is above the original PO price, but the increase may be "
                "legitimate. The agent must compare the PO, request the support packet, confirm the goods "
                "receipt, and approve only if the amended charge is properly documented."
            ),
            max_steps=8,
            supplier=SupplierProfile(
                supplier_id="SUP-410",
                supplier_name="Apex Cold Chain Logistics",
                risk_level="medium",
                payment_terms="Net 20",
                notes="Temperature-controlled lanes allow revised rates only with signed change approval.",
            ),
            invoice=Invoice(
                invoice_id="INV-8814",
                supplier_id="SUP-410",
                amount=5520.0,
                currency="USD",
                quantity_billed=12,
                unit_price=460.0,
                due_date="2026-04-27",
                po_number="PO-1107",
                header_note="Revised refrigerated route charge applied after weekend emergency dispatch.",
            ),
            purchase_order=PurchaseOrder(
                po_number="PO-1107",
                sku="COLD-ROUTE-3",
                quantity=12,
                unit_price=400.0,
                currency="USD",
                approved_change_note=(
                    "Signed change order CO-118 approved on 2026-04-06 authorizes 460 USD per lane "
                    "for emergency cold-chain coverage."
                ),
            ),
            goods_receipt=GoodsReceipt(
                receipt_id="GR-5108",
                quantity_received=12,
                received_on="2026-04-07",
                warehouse_notes="Emergency cold-chain delivery completed in full with no spoilage.",
            ),
            duplicate_invoice_found=False,
            relevant_actions=["review_invoice", "compare_po", "request_supporting_doc", "check_receipt"],
            required_evidence=["reviewed_invoice", "checked_po", "requested_doc", "checked_receipt"],
            nice_to_have_evidence=[],
            final_action="approve",
            risk_focus="Do not escalate a valid exception if the approval trail and receipt are both present.",
            resolution_justification="Invoice approved because the higher rate is covered by the signed change order and the delivery was received in full.",
            scoring_weights={
                "reviewed_invoice": 0.08,
                "checked_po": 0.18,
                "requested_doc": 0.22,
                "checked_receipt": 0.14,
                "resolved_correctly": 0.28,
                "closed_case": 0.09,
            },
        ),
    }


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def clamp_strict_score(value: float) -> float:
    return max(MIN_STRICT_SCORE, min(MAX_STRICT_SCORE, round(value, 4)))


def round_amount(value: float) -> float:
    return round(value, 4)


def build_public_task(task: TaskDefinition) -> Dict[str, Any]:
    return {
        "task_id": task.task_id,
        "name": task.name,
        "difficulty": task.difficulty,
        "description": task.description,
        "max_steps": task.max_steps,
    }


def current_observation(episode: Episode) -> Dict[str, Any]:
    task = episode.task
    docs: Dict[str, Any] = {"invoice": asdict(task.invoice)}
    if episode.revealed_docs["purchase_order"]:
        docs["purchase_order"] = asdict(task.purchase_order)
    if episode.revealed_docs["goods_receipt"]:
        docs["goods_receipt"] = asdict(task.goods_receipt)
    if episode.revealed_docs["duplicate_search"]:
        docs["duplicate_search"] = {
            "duplicate_found": task.duplicate_invoice_found,
            "matching_invoice_id": task.invoice.duplicate_reference,
            "search_note": (
                "A prior invoice with the same PO, quantity, and amount has already been paid."
                if task.duplicate_invoice_found
                else "No paid invoice with the same commercial fingerprint was found."
            ),
        }
    if episode.revealed_docs["supporting_doc"]:
        docs["supporting_doc"] = {
            "status": "received" if task.purchase_order.approved_change_note else "missing",
            "note": task.purchase_order.approved_change_note
            or "No approved surcharge or change-order document is available.",
        }
    return {
        "session_id": episode.session_id,
        "task_id": task.task_id,
        "task_name": task.name,
        "task_description": task.description,
        "step_count": episode.step_count,
        "max_steps": task.max_steps,
        "available_actions": AVAILABLE_ACTIONS,
        "risk_focus": task.risk_focus,
        "supplier_profile": asdict(task.supplier),
        "documents": docs,
        "milestones": deepcopy(episode.milestones),
        "final_decision": episode.final_decision,
        "last_action_result": episode.last_action_result,
        "last_action_error": episode.last_action_error,
    }


def episode_state(episode: Episode) -> Dict[str, Any]:
    return {
        "environment": ENV_NAME,
        "session_id": episode.session_id,
        "done": episode.done,
        "step_count": episode.step_count,
        "max_steps": episode.task.max_steps,
        "score": clamp_strict_score(score_episode(episode)),
        "total_reward": round_amount(episode.total_reward),
        "reward_history": episode.reward_history,
        "action_history": episode.action_history,
        "observation": current_observation(episode),
    }


def score_episode(episode: Episode) -> float:
    positive = 0.0
    for key, weight in episode.task.scoring_weights.items():
        if episode.milestones.get(key):
            positive += weight
    penalty = min(0.35, (episode.unsafe_resolution_attempts * 0.08) + (episode.irrelevant_actions * 0.03))
    if episode.done and episode.final_decision is None:
        penalty += 0.04
    return clamp_strict_score(positive - penalty)


def reset_episode(task_id: str, session_id: str) -> Episode:
    return Episode(session_id=session_id, task=TASKS[task_id])


def describe_missing_evidence(episode: Episode) -> List[str]:
    return [
        EVIDENCE_LABELS[key]
        for key in episode.task.required_evidence
        if not episode.milestones.get(key, False)
    ]


def resolve_session_id(explicit_session_id: Optional[str]) -> Tuple[Optional[Episode], Optional[str]]:
    if explicit_session_id and explicit_session_id in SESSIONS:
        return SESSIONS[explicit_session_id], explicit_session_id
    if not explicit_session_id:
        if LAST_SESSION_ID and LAST_SESSION_ID in SESSIONS:
            return SESSIONS[LAST_SESSION_ID], LAST_SESSION_ID
        if len(SESSIONS) == 1:
            only_session = next(iter(SESSIONS))
            return SESSIONS[only_session], only_session
    return None, explicit_session_id


def step_episode(episode: Episode, action_type: str, rationale: str = "") -> Dict[str, Any]:
    if episode.done:
        episode.last_action_error = "Episode is already complete."
        return {
            "session_id": episode.session_id,
            "observation": current_observation(episode),
            "reward": 0.0,
            "done": True,
            "score": clamp_strict_score(score_episode(episode)),
            "info": {"error": episode.last_action_error},
        }

    episode.step_count += 1
    episode.last_action_error = None
    task = episode.task
    reward = 0.03 if action_type in task.relevant_actions or action_type in ("approve","route_to_review","reject","done") else 0.0
    info: Dict[str, Any] = {"rationale": rationale}

    def mark_evidence(
        milestone: str,
        amount: float,
        success_message: str,
        repeat_message: str,
        irrelevant_message: str,
        reveal_key: Optional[str] = None,
    ) -> None:
        nonlocal reward
        if action_type not in task.relevant_actions:
            episode.irrelevant_actions += 1
            reward = max(0.0, reward - 0.02)
            episode.last_action_result = irrelevant_message
            return
        if reveal_key:
            episode.revealed_docs[reveal_key] = True
        if episode.milestones[milestone]:
            episode.irrelevant_actions += 1
            reward = max(0.0, reward - 0.01)
            episode.last_action_result = repeat_message
            return
        episode.milestones[milestone] = True
        reward += amount
        episode.last_action_result = success_message

    def register_resolution(success_amount: float, success_message: str) -> None:
        nonlocal reward
        episode.final_decision = action_type
        missing = describe_missing_evidence(episode)
        if action_type != task.final_action:
            episode.unsafe_resolution_attempts += 1
            reward = max(0.0, reward - 0.03)
            episode.last_action_error = (
                f"Incorrect resolution for this case. The evidence does not support '{action_type}'."
            )
            episode.last_action_result = "Resolution attempt did not match the case facts."
            info["missing_evidence"] = missing
            return
        if missing:
            episode.unsafe_resolution_attempts += 1
            reward = max(0.0, reward - 0.05)
            episode.last_action_error = "Decision lacks required evidence: " + ", ".join(missing) + "."
            episode.last_action_result = "Resolution was attempted before the case file was complete."
            info["missing_evidence"] = missing
            return
        bonus = 0.03 * sum(1 for item in task.nice_to_have_evidence if episode.milestones.get(item))
        episode.milestones["resolved_correctly"] = True
        episode.last_action_result = success_message
        reward += success_amount + bonus
        info["missing_evidence"] = []

    if action_type == "review_invoice":
        mark_evidence(
            "reviewed_invoice",
            0.14,
            "Invoice header reviewed for amount, supplier, due date, and exception signal.",
            "Invoice header was already reviewed earlier in the case.",
            "Re-reading the invoice did not add new evidence for this case.",
        )
    elif action_type == "compare_po":
        mark_evidence(
            "checked_po",
            0.18,
            "Purchase order opened for price and quantity comparison.",
            "Purchase order has already been compared in this session.",
            "PO comparison was not a priority for this case at this stage.",
            reveal_key="purchase_order",
        )
    elif action_type == "check_receipt":
        mark_evidence(
            "checked_receipt",
            0.16,
            "Goods receipt pulled in to confirm operational delivery status.",
            "Goods receipt was already checked in this session.",
            "Receipt verification did not add much value for this case.",
            reveal_key="goods_receipt",
        )
    elif action_type == "search_duplicate":
        mark_evidence(
            "checked_duplicates",
            0.22 if task.duplicate_invoice_found else 0.08,
            "Duplicate search completed across recent invoice history.",
            "Duplicate search was already run for this session.",
            "Duplicate search was not strongly indicated for this case.",
            reveal_key="duplicate_search",
        )
    elif action_type == "request_supporting_doc":
        mark_evidence(
            "requested_doc",
            0.2 if task.purchase_order.approved_change_note else 0.18,
            "Supporting documentation packet attached to the case file.",
            "Supporting documents were already requested earlier.",
            "Requesting support was not essential for this case.",
            reveal_key="supporting_doc",
        )
    elif action_type == "flag_duplicate":
        if "flag_duplicate" not in task.relevant_actions:
            episode.irrelevant_actions += 1
            reward = max(0.0, reward - 0.02)
            episode.last_action_result = "Duplicate flagging was not a meaningful control for this case."
        elif not episode.milestones["checked_duplicates"]:
            episode.unsafe_resolution_attempts += 1
            reward = max(0.0, reward - 0.03)
            episode.last_action_error = "Run a duplicate search before escalating a duplicate risk."
        elif episode.milestones["flagged_risk"]:
            episode.irrelevant_actions += 1
            reward = max(0.0, reward - 0.01)
            episode.last_action_result = "Duplicate risk was already flagged earlier in the workflow."
        else:
            episode.milestones["flagged_risk"] = True
            reward += 0.09
            episode.last_action_result = "Potential duplicate escalated to finance control for review."
    elif action_type == "approve":
        register_resolution(0.3, task.resolution_justification)
    elif action_type == "route_to_review":
        register_resolution(0.28, task.resolution_justification)
    elif action_type == "reject":
        register_resolution(0.29, task.resolution_justification)
    elif action_type == "done":
        if episode.milestones["resolved_correctly"]:
            if episode.milestones["closed_case"]:
                episode.irrelevant_actions += 1
                reward = max(0.0, reward - 0.01)
                episode.last_action_result = "Case was already closed."
            else:
                episode.milestones["closed_case"] = True
                reward += 0.1
                episode.last_action_result = "Case closed with a documented evidence trail."
        else:
            episode.unsafe_resolution_attempts += 1
            reward = max(0.0, reward - 0.02)
            if episode.final_decision:
                episode.last_action_error = "Case was closed with an unsupported or incorrect resolution."
            else:
                episode.last_action_error = "Case cannot close before a final evidence-backed decision."
            episode.last_action_result = "Case closed without a valid resolution."
        episode.done = True
    else:
        episode.unsafe_resolution_attempts += 1
        reward = max(0.0, reward - 0.03)
        episode.last_action_error = f"Unsupported action_type '{action_type}'."

    if episode.step_count >= task.max_steps and not episode.done:
        episode.done = True
        episode.last_action_error = episode.last_action_error or "Maximum steps reached before supported closure."
        episode.last_action_result = episode.last_action_result or "Maximum steps reached; case auto-closed."

    reward = clamp_unit(reward)
    episode.total_reward = round_amount(episode.total_reward + reward)
    episode.reward_history.append(reward)
    episode.action_history.append(
        {
            "step": episode.step_count,
            "action_type": action_type,
            "rationale": rationale,
            "reward": reward,
            "final_decision": episode.final_decision,
        }
    )

    return {
        "session_id": episode.session_id,
        "observation": current_observation(episode),
        "reward": reward,
        "done": episode.done,
        "score": clamp_strict_score(score_episode(episode)),
        "info": info,
    }


class OpenEnvHandler(BaseHTTPRequestHandler):
    server_version = "OpenEnvInvoice/2.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/":
            self._send_json({"name": ENV_NAME, "status": "ok", "message": "Invoice Exception OpenEnv is running."})
            return
        if path == "/health":
            self._send_json({"status": "ok"})
            return
        if path == "/metadata":
            self._send_json(
                {
                    "name": ENV_NAME,
                    "version": "2.0.0",
                    "task_count": len(TASKS),
                    "tasks": [build_public_task(task) for task in TASKS.values()],
                }
            )
            return
        if path == "/tasks":
            self._send_json({"tasks": [build_public_task(task) for task in TASKS.values()]})
            return
        if path == "/state":
            session_id = query.get("session_id", [None])[0]
            with SESSION_LOCK:
                episode, resolved_session_id = resolve_session_id(session_id)
                if episode is None:
                    self._send_json(
                        {
                            "error": "Unknown session. Call /reset first and pass session_id to /state.",
                            "session_id": resolved_session_id,
                        },
                        HTTPStatus.BAD_REQUEST,
                    )
                    return
                self._send_json(episode_state(episode))
            return
        self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        global LAST_SESSION_ID

        parsed = urlparse(self.path)
        path = parsed.path
        payload = self._parse_json_body()

        if path == "/reset":
            task_id = payload.get("task_id", "invoice_easy")
            if task_id not in TASKS:
                self._send_json({"error": f"Unknown task_id '{task_id}'."}, HTTPStatus.BAD_REQUEST)
                return
            session_id = str(payload.get("session_id") or uuid4())
            episode = reset_episode(task_id, session_id)
            with SESSION_LOCK:
                SESSIONS[session_id] = episode
                LAST_SESSION_ID = session_id
            self._send_json(
                {
                    "session_id": session_id,
                    "task": build_public_task(episode.task),
                    "observation": current_observation(episode),
                    "done": episode.done,
                    "score": clamp_strict_score(score_episode(episode)),
                }
            )
            return

        if path == "/step":
            with SESSION_LOCK:
                episode, resolved_session_id = resolve_session_id(payload.get("session_id"))
                if episode is None:
                    self._send_json(
                        {
                            "error": "Unknown session. Call /reset first and send session_id with /step.",
                            "session_id": resolved_session_id,
                        },
                        HTTPStatus.BAD_REQUEST,
                    )
                    return
                result = step_episode(episode, payload.get("action_type", ""), payload.get("rationale", ""))
                LAST_SESSION_ID = episode.session_id
            self._send_json(result)
            return

        self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _parse_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    global TASKS
    TASKS = build_tasks()
    server = ThreadingHTTPServer(("0.0.0.0", DEFAULT_PORT), OpenEnvHandler)
    print(f"{ENV_NAME} listening on http://0.0.0.0:{DEFAULT_PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
