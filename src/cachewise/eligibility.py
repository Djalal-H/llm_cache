"""Conservative FAQ routing; never use dataset labels at runtime.

Future router option: TypeSafe AI's Jev model is a good fit for replacing the
lexical classifier below. Jev accepts a state plus typed questions and returns
structured decisions, probabilities, and confidence instead of generated text.
See ``JEV_INTEGRATION`` below for the intended boundary. The current release
does not call Jev and remains deterministic and dependency-free.
"""

import re
import unicodedata

from cachewise.models import ChatRequest


def normalize_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFC", question).split())


# Exclusions take precedence over topic matches. First-person policy wording such as
# 'What is my return window?' is safe; specific purchases and account actions are not.
PERSONAL = re.compile(
    r"\bcw\s*[-–]?\s*\d+\b|\border\b|\b(?:tracking|refund|account|password|charged|paid|"
    r"invoice|receipt|balance|address|status|arrive|arriving|arrived|yesterday|today|"
    r"tomorrow|last week|last month)\b|"
    r"\b(?:my|our|this|that|the)\s+(?:\w+\s+){0,2}"
    r"(?:order|purchase|shipment|package|parcel|payment|item|product)\b|"
    r"\bi\s+(?:have(?! to return an? item)|had|bought|ordered|received|"
    r"purchased|returned|need|want)\b",
    re.I,
)
UNSAFE = re.compile(
    r"\b(?:stock|inventory|sold out|available now|availability|ignore|instructions?|"
    r"prompt|pretend|instead|override|translate|summari[sz]e|repeat|say|write|"
    r"remember|forget|system|assistant|secret|name|email|phone|weather|joke)\b|"
    r"[`{}<>]|https?://|@",
    re.I,
)
TOPICS = (
    re.compile(r"\b(?:returns?|returnable|returning|postage|send back)\b", re.I),
    re.compile(r"\b(?:shipping|ship|shipped|delivery|deliver|courier)\b", re.I),
    re.compile(
        r"\b(?:payment|pay|checkout|visa|mastercard|paypal|bitcoin|crypto\w*|"
        r"ethereum|bank transfer|wire transfer)\b",
        re.I,
    ),
)
POLICY_INTENT = re.compile(
    r"^(?:what|which|how|when|where|can|could|do|does|is|are|may|will|"
    r"tell me|explain|list|give me|please (?:tell|explain|list))\b|"
    r"^(?:returns?|shipping|payment methods?)\??$",
    re.I,
)
# Conjunctions introduce unclassified extra work; allow the few common policy phrases.
SAFE_CONJUNCTIONS = re.compile(
    r"\b(?:opened and used|credit and debit|terms and conditions)\b", re.I
)


# Vocabulary constrains the entire utterance, not just the matching keyword.
# New domain wording must be reviewed before it can enter the shared cache.
FAQ_WORDS = frozenset(
    """
a an the i my me we our you your it its they their this that these those
what which how when where can could do does is are may will would should
have to for from in on at with without by of as be been if not no any all
and or only than more less must need require required accepted accept supported
support available option options method methods payment payments pay paying
purchase purchases checkout cash credit debit card cards visa mastercard paypal
bitcoin cryptocurrency crypto ethereum bank wire transfer transfers courier
returns return returnable returning returned send back window windows days day
calendar business deadline long many item items product products personalized
custom engraved defective faulty used unused opened original packaging condition
conditions rule rules policy policies exception exceptions dislike fee fees
label labels postage shipping ship ships shipped order delivery deliver standard usual duration
time times cost costs charge charges free minimum spend spending premium customers
customer region regional us united states eu european union outside international
algeria canada japan brazil anywhere destination destinations dollar dollars euro euros
much amount take takes get explain tell list give please about does need shipping
how paid allow allowed eligible eligibility options deadline returning purchase
""".split()
)


# FUTURE(Jev): replace only the classification section of ``eligibility_reason``
# with an injected async eligibility provider; keep normalization and the
# cache-safety decision in application code. A Jev Choice question should use
# the stable outcomes below so metrics and bypass behavior remain compatible:
#
#   eligible          shared returns/shipping/payment policy FAQ
#   personalized      needs customer, order, payment, tracking, or refund data
#   unsafe_or_live    live stock or instruction-manipulation request
#   mixed_or_uncertain multiple intents, unsupported topic, or unclear wording
#
# Send the normalized question as ``state`` to ``jev-latest``. Only convert a
# high-confidence ``eligible`` choice into cache eligibility; every other
# choice, low-confidence result, timeout, invalid response, or provider failure
# must fail closed to ``mixed_or_uncertain``. Before enabling it, freeze a
# confidence threshold on the tuning split and validate false-cache decisions
# on held-out/adversarial traffic. Do not silently fall back to the permissive
# outcome. TypeSafe API: POST https://api.typesafe.ai/v1/systemone.
JEV_INTEGRATION = "planned"


def eligibility_reason(request: ChatRequest) -> str:
    """Return 'eligible' or a stable bypass reason. Unknown wording fails closed."""
    if request.history:
        return "history"
    question = normalize_question(request.question)
    if UNSAFE.search(request.question):
        return "unsafe_or_live"
    personal_question = re.sub(
        r"\bcan i have an order shipped to\b", "can delivery to", question, flags=re.I
    )
    if PERSONAL.search(personal_question):
        return "personalized"
    if len(question) > 240 or re.search(r"[;:!]|\?\s*\S|\.\s+\S", question):
        return "mixed_or_uncertain"
    remaining = SAFE_CONJUNCTIONS.sub("", question)
    if re.search(r"\b(?:and|also|then|but|plus|as well|while)\b", remaining, re.I):
        return "mixed_or_uncertain"
    words = re.findall(r"[^\W\d_]+", question.casefold())
    if any(word not in FAQ_WORDS for word in words) or not POLICY_INTENT.search(question):
        return "unrecognized_wording"
    if not any(topic.search(question) for topic in TOPICS):
        return "unknown_topic"
    return "eligible"
