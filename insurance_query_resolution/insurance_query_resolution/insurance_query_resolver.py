"""
Intelligent Insurance Query Resolution - Base Project
-----------------------------------------------------
Pipeline:
  1. Preprocess the customer query
  2. Extract entities (policy number, claim ID)
  3. Detect intent (TF-IDF similarity against example phrases)
  4. Resolve: account lookups (mock DB) or FAQ retrieval (TF-IDF)
  5. Escalate to a human agent if confidence is low or the customer is upset

Install:  pip install scikit-learn
Run:      python insurance_query_resolver.py
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# 1. Mock data (replace with a real database / API later)
# --------------------------------------------------------------------------
POLICIES = {
    "POL1001": {"holder": "Asha Rao", "type": "Health", "premium": 12500,
                "due_date": "2026-12-05", "status": "Active", "cover": 500000},
    "POL1002": {"holder": "Ravi Kumar", "type": "Motor", "premium": 8200,
                "due_date": "2026-11-20", "status": "Active", "cover": 800000},
}

CLAIMS = {
    "CLM2001": {"policy": "POL1001", "status": "Under review",
                "amount": 45000, "updated": "2026-11-06"},
    "CLM2002": {"policy": "POL1002", "status": "Approved",
                "amount": 18000, "updated": "2026-11-03"},
}

FAQS = [
    ("How do I file a claim?",
     "Log in to the portal or call our helpline, submit the claim form with "
     "your policy number, ID proof, and supporting documents (bills, FIR, "
     "photos). You will get a claim ID by SMS/email."),
    ("What documents are required for a health claim?",
     "Claim form, hospital discharge summary, original bills and receipts, "
     "doctor's prescription, diagnostic reports, and your ID proof."),
    ("How can I renew my policy?",
     "You can renew online through the portal, via the mobile app, or at any "
     "branch. Renew before the due date to avoid a lapse in cover."),
    ("What happens if I miss a premium payment?",
     "Most policies include a 30-day grace period. After that the policy may "
     "lapse, but it can usually be revived with a late fee."),
    ("How do I cancel my policy?",
     "Send a written cancellation request with your policy number. A refund, "
     "if applicable, is calculated on a pro-rata basis per policy terms."),
    ("What is a deductible?",
     "A deductible is the amount you pay out of pocket before the insurer "
     "starts paying for a covered claim."),
    ("What is the waiting period?",
     "A waiting period is the time after buying a policy during which certain "
     "benefits (e.g., pre-existing conditions) cannot be claimed."),
    ("How do I update my address or contact details?",
     "Submit an update request on the portal with a valid address proof, or "
     "visit your nearest branch."),
]

# Example phrases per intent, used for TF-IDF intent matching
INTENTS = {
    "claim_status": ["what is the status of my claim", "track claim",
                     "claim progress update", "has my claim been approved"],
    "policy_details": ["show my policy details", "what is my coverage",
                       "policy information", "sum insured of my policy"],
    "premium_due": ["when is my premium due", "premium amount",
                    "next payment date", "how much do i have to pay"],
    "faq": [q for q, _ in FAQS],
}

ESCALATION_WORDS = {"angry", "fraud", "complaint", "lawyer", "legal", "worst",
                    "terrible", "unacceptable", "urgent", "ombudsman", "scam"}

CONFIDENCE_THRESHOLD = 0.20


# --------------------------------------------------------------------------
# 2. Helper functions
# --------------------------------------------------------------------------
def clean(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


def extract_entities(text: str) -> dict:
    policy = re.search(r"\bPOL\d{4}\b", text, re.I)
    claim = re.search(r"\bCLM\d{4}\b", text, re.I)
    return {
        "policy_id": policy.group().upper() if policy else None,
        "claim_id": claim.group().upper() if claim else None,
    }


# --------------------------------------------------------------------------
# 3. Resolver
# --------------------------------------------------------------------------
class InsuranceQueryResolver:
    def __init__(self):
        # Intent classifier
        self.intent_labels, intent_docs = [], []
        for label, phrases in INTENTS.items():
            for p in phrases:
                self.intent_labels.append(label)
                intent_docs.append(clean(p))
        self.intent_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.intent_matrix = self.intent_vec.fit_transform(intent_docs)

        # FAQ retriever
        self.faq_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.faq_matrix = self.faq_vec.fit_transform(
            [clean(q + " " + a) for q, a in FAQS])

    def detect_intent(self, query: str):
        sims = cosine_similarity(self.intent_vec.transform([clean(query)]),
                                 self.intent_matrix)[0]
        best = sims.argmax()
        return self.intent_labels[best], float(sims[best])

    def answer_faq(self, query: str):
        sims = cosine_similarity(self.faq_vec.transform([clean(query)]),
                                 self.faq_matrix)[0]
        best = sims.argmax()
        return FAQS[best][1], float(sims[best])

    def resolve(self, query: str) -> dict:
        words = set(clean(query).split())
        entities = extract_entities(query)

        # Escalate upset / sensitive customers right away
        if words & ESCALATION_WORDS:
            return self._result("escalation", 1.0,
                "I'm sorry for the trouble. I'm connecting you to a human "
                "agent who will help you right away.", escalate=True)

        intent, conf = self.detect_intent(query)
        if conf < CONFIDENCE_THRESHOLD:
            intent = "faq"  # weak intent match -> try FAQ retrieval

        if intent == "claim_status":
            cid = entities["claim_id"]
            if not cid:
                return self._result(intent, conf,
                    "Please share your claim ID (e.g., CLM2001).")
            claim = CLAIMS.get(cid)
            if not claim:
                return self._result(intent, conf, f"I couldn't find claim {cid}.")
            return self._result(intent, conf,
                f"Claim {cid} is '{claim['status']}' (amount Rs.{claim['amount']:,}, "
                f"last updated {claim['updated']}).")

        if intent in ("policy_details", "premium_due"):
            pid = entities["policy_id"]
            if not pid:
                return self._result(intent, conf,
                    "Please share your policy number (e.g., POL1001).")
            pol = POLICIES.get(pid)
            if not pol:
                return self._result(intent, conf, f"I couldn't find policy {pid}.")
            if intent == "premium_due":
                return self._result(intent, conf,
                    f"Your {pol['type']} premium of Rs.{pol['premium']:,} "
                    f"is due on {pol['due_date']}.")
            return self._result(intent, conf,
                f"Policy {pid}: {pol['type']} insurance for {pol['holder']}, "
                f"status {pol['status']}, cover Rs.{pol['cover']:,}.")

        # Fallback: FAQ retrieval
        answer, score = self.answer_faq(query)
        if score < CONFIDENCE_THRESHOLD:
            return self._result("unknown", score,
                "I'm not sure I understood. Let me connect you to a human agent.",
                escalate=True)
        return self._result("faq", score, answer)

    @staticmethod
    def _result(intent, confidence, response, escalate=False):
        return {"intent": intent, "confidence": round(confidence, 2),
                "response": response, "escalate": escalate}


# --------------------------------------------------------------------------
# 4. Command-line chat loop
# --------------------------------------------------------------------------
if __name__ == "__main__":
    bot = InsuranceQueryResolver()
    print("Insurance Assistant (type 'quit' to exit)")
    print("Try: 'What is the status of claim CLM2001?'  |  'How do I file a claim?'\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        if not user:
            continue
        r = bot.resolve(user)
        print(f"Bot: {r['response']}")
        print(f"     [intent={r['intent']}, confidence={r['confidence']}, "
              f"escalate={r['escalate']}]\n")
