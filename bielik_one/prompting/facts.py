"""Deterministic facts: calendar and number arithmetic computed from a request and appended to its state.

A decision model answers in one forward pass, so it is weak at exact arithmetic: counting the days between two dates,
adding 30 days or 10 working days to a date, adding up the amounts of a list, or comparing an amount with a threshold
stated in the rule. With `"facts": true` the API computes these facts in code and appends them to the state under a
short header, so the model reads the result instead of computing it. Code computes, the model decides:

- only arithmetic and calendar facts are added, never a rule or a verdict ("the deadline is met", "the limit is
  exceeded"); which fact matters, and how the rule counts (inclusive or not, which days are excluded), stays the
  model's decision;
- comparisons are symbols only (`>`, `<`, `=`);
- the state text is never changed, only extended.

What is computed (Polish, English, German and Norwegian Bokmål text; the facts are written in the request language):

- every date of the state with its weekday, and its distance in days from today when the state gives today's date;
- the days between the other dates of the state (also counting both days, and in working days Monday-Friday without
  the days off the text names, when the text talks about working days); full months and years for longer spans;
- date + duration for the durations in the instructions (and the state): "30 days", "10 working days", "2 weeks",
  "6 months", "3 years", "the 14th day";
- clock time between dated times ("from 23:10 on 8 February to 01:50 on 9 February: 2 h 40 min");
- month ranges such as "09.2019 - 08.2021" or "03.2020 - present", in months, and their total without overlaps;
- sums of amounts with the same unit (currency, %, MB, kg, hours, ...) within one sentence, of a list, and of the list
  lines that share a description; an amount per person when the rule speaks of an amount per person;
- comparisons of these amounts with the thresholds of the same unit in the instructions;
- whether the state contains a bank account number, an e-mail address or a postal code, when the instructions ask for
  one.

  from bielik_one.prompting.facts import facts, inject
  lines = facts(state, ["Was the appeal filed within 14 days of delivery?"])
  state2 = inject(state, lines, language)
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from itertools import islice

LANGUAGES = ("pl", "en", "de", "nb")

# Category caps keep the block short: a long block of facts distracts more than it helps.
MAX_DATES, MAX_GAPS, MAX_PLUS, MAX_TODAY_PLUS = 6, 5, 6, 4
MAX_HOURS, MAX_MONTH_RANGES = 2, 4
MAX_SUMS, MAX_PER_PERSON, MAX_COMPARE, MAX_IDS = 4, 2, 6, 3
MAX_DURATIONS, MAX_ANCHORS = 3, 3
MAX_CALENDAR = 14  # all date lines together (dates, gaps, clock time, date + duration)
# The API takes untrusted text: facts read at most this much of the state and of the instructions, and collect at
# most MAX_MATCHES dates, times and amounts, so the work per request stays small and linear.
MAX_INPUT_CHARS = 20_000
MAX_MATCHES = 300
# Days off: at most this many, for year-less dates ("1 May") at most MAX_YEARS years (those of the dates of the text).
MAX_HOLIDAYS, MAX_YEARS = 50, 3

# ------------------------------------------------------------------ words per language

HEADER = {
    "pl": "Fakty wyliczone automatycznie z tekstu (arytmetyka i kalendarz, bez oceny):",
    "en": "Facts computed automatically from the text (arithmetic and calendar, no judgement):",
    "de": "Automatisch aus dem Text berechnete Fakten (Arithmetik und Kalender, ohne Bewertung):",
    "nb": "Fakta beregnet automatisk fra teksten (aritmetikk og kalender, uten vurdering):",
}
WEEKDAYS = {
    "pl": ["poniedziałek", "wtorek", "środa", "czwartek", "piątek", "sobota", "niedziela"],
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "de": ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"],
    "nb": ["mandag", "tirsdag", "onsdag", "torsdag", "fredag", "lørdag", "søndag"],
}
# unit -> (singular, plural); "wd" = working days
UNIT_WORDS = {
    "pl": {"d": ("dzień", "dni"), "wd": ("dzień roboczy", "dni roboczych"), "w": ("tydzień", "tyg."),
           "m": ("miesiąc", "mies."), "y": ("rok", "lat")},
    "en": {"d": ("day", "days"), "wd": ("working day", "working days"), "w": ("week", "weeks"),
           "m": ("month", "months"), "y": ("year", "years")},
    "de": {"d": ("Tag", "Tage"), "wd": ("Werktag", "Werktage"), "w": ("Woche", "Wochen"),
           "m": ("Monat", "Monate"), "y": ("Jahr", "Jahre")},
    "nb": {"d": ("dag", "dager"), "wd": ("virkedag", "virkedager"), "w": ("uke", "uker"),
           "m": ("måned", "måneder"), "y": ("år", "år")},
}
T = {
    "pl": dict(
        holiday="dzień wolny według tekstu", today="dzisiaj", before="{n} {u} przed dzisiejszą datą", after="{n} {u} po dzisiejszej dacie",
        gap="Od {a} do {b}: {n} {u} ({n1} licząc oba dni)", workdays="dni robocze (pn-pt{hol}) od dnia po {a} do {b} włącznie: {w}",
        hol=", bez dni wolnych podanych w tekście", span="pełne lata: {y} (+ pełne miesiące: {m})",
        months="pełne miesiące: {n}",
        hours="Od {a} do {b}: {h} godz. {m} min", mrange="{a} - {b}: {n} mies. (licząc oba miesiące)",
        mtotal="Okresy razem (bez podwójnego liczenia): {n} mies. = {y} lat i {m} mies.",
        per_person="{a} ÷ {n} osób = {r}", found="W tekście jest {what}: {value}",
        missing="W tekście nie ma: {what}",
        what=dict(account="numer rachunku bankowego", email="adres e-mail", postal="kod pocztowy")),
    "en": dict(
        holiday="a day off named in the text", today="today", before="{n} {u} before today", after="{n} {u} after today",
        gap="From {a} to {b}: {n} {u} ({n1} counting both days)", workdays="working days (Mon-Fri{hol}) from the day after {a} to {b} inclusive: {w}",
        hol=", without the days off named in the text", span="full years: {y} (+ full months: {m})",
        months="full months: {n}",
        hours="From {a} to {b}: {h} h {m} min", mrange="{a} - {b}: {n} months (counting both months)",
        mtotal="Periods in total (overlaps counted once): {n} months = {y} years and {m} months",
        per_person="{a} ÷ {n} persons = {r}", found="The text contains {what}: {value}",
        missing="The text contains no {what}",
        what=dict(account="bank account number", email="e-mail address", postal="postcode")),
    "de": dict(
        holiday="im Text genannter freier Tag", today="heute", before="{n} {u} vor dem heutigen Datum", after="{n} {u} nach dem heutigen Datum",
        gap="Vom {a} bis {b}: {n} {u} ({n1} mit beiden Tagen)", workdays="Werktage (Mo-Fr{hol}) vom Tag nach dem {a} bis einschließlich {b}: {w}",
        hol=", ohne die im Text genannten freien Tage", span="volle Jahre: {y} (+ volle Monate: {m})",
        months="volle Monate: {n}",
        hours="Von {a} bis {b}: {h} Std. {m} Min.", mrange="{a} - {b}: {n} Monate (beide Monate mitgezählt)",
        mtotal="Zeiträume insgesamt (Überschneidungen einmal gezählt): {n} Monate = {y} Jahre und {m} Monate",
        per_person="{a} ÷ {n} Personen = {r}", found="Der Text enthält {what}: {value}",
        missing="Der Text enthält keine {what}",
        what=dict(account="Kontonummer (IBAN)", email="E-Mail-Adresse", postal="Postleitzahl")),
    "nb": dict(
        holiday="fridag nevnt i teksten", today="i dag", before="{n} {u} før dagens dato", after="{n} {u} etter dagens dato",
        gap="Fra {a} til {b}: {n} {u} ({n1} med begge dagene)", workdays="virkedager (man-fre{hol}) fra dagen etter {a} til og med {b}: {w}",
        hol=", uten fridagene som er nevnt i teksten", span="hele år: {y} (+ hele måneder: {m})",
        months="hele måneder: {n}",
        hours="Fra {a} til {b}: {h} t {m} min", mrange="{a} - {b}: {n} måneder (begge månedene medregnet)",
        mtotal="Periodene til sammen (overlapp telt én gang): {n} måneder = {y} år og {m} måneder",
        per_person="{a} ÷ {n} personer = {r}", found="Teksten inneholder {what}: {value}",
        missing="Teksten inneholder ikke {what}",
        what=dict(account="kontonummer", email="e-postadresse", postal="postnummer")),
}

# Language detection: frequent short words (a request is long enough for this to be reliable).
STOPWORDS = {
    "pl": "i w na się jest nie z do że oraz lub od po przez dla jeżeli jeśli czy dni który która które został".split(),
    "en": "the and is of to in for on that with was are by it or not if from has have be days which".split(),
    "de": "der die das und ist nicht den dem des mit von zu für auf ein eine wenn oder wird werden Tage".split(),
    "nb": "og er det ikke til på som for av med en et har den de om skal dager eller ble fra".split(),
}

MONTHS = {}
for _names in (
    "stycznia lutego marca kwietnia maja czerwca lipca sierpnia września października listopada grudnia",
    "january february march april may june july august september october november december",
    "januar februar märz april mai juni juli august september oktober november dezember",
    "januar februar mars april mai juni juli august september oktober november desember",
    "jan feb mar apr may jun jul aug sep oct nov dec",
):
    for _i, _n in enumerate(_names.split()):
        MONTHS.setdefault(_n, _i + 1)
MONTHS.update({"sept": 9, "maerz": 3, "mär": 3, "okt": 10, "dez": 12, "des": 12, "mai": 5})
# Norwegian "des" only as "des.": a bare "des" is the German article ("Artikel 5 des 2020 erlassenen Gesetzes").
_MONTH_RX = "|".join(sorted((n for n in MONTHS if n != "des"), key=len, reverse=True)) + r"|des(?=\.)"

DATE_TXT = re.compile(r"(?<![\w.])(\d{1,2})(?:\.|st|nd|rd|th)?\s+(" + _MONTH_RX + r")\.?,?\s+(\d{4})(?!\d)", re.I)
DATE_US = re.compile(r"\b(" + _MONTH_RX + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})(?!\d)", re.I)
DATE_ISO = re.compile(r"(?<![\d.\-/])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d\-])")
DATE_DMY = re.compile(r"(?<![\d.\-/])(\d{1,2})([./])(\d{1,2})\2(\d{4})(?![\d.]\d)")
DATE_NO_YEAR = re.compile(r"(?<![\w.])(\d{1,2})(?:\.|st|nd|rd|th)?\s+(" + _MONTH_RX + r")\b(?!\.?,?\s+\d{4})", re.I)
TIME = re.compile(r"(?<![\d:.])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])")
TIME_BEFORE = re.compile(r"^[\s,]*(?:on|am|den|w|på|the|,|\s)*(?:[A-Za-zÀ-ž]+day|[a-zA-ZÀ-ž]+tag|[a-zæøåA-ZÆØÅ]+dag|"
                         r"poniedziałek|wtorek|środ[aę]|czwartek|piątek|sobot[aę]|niedziel[aę])?,?\s*$")
TIME_AFTER = re.compile(r"^[\s,]*(?:r\.)?[\s,]*(?:o|at|um|kl\.?|klokka|klokken|godz\.|ok\.|około|ca\.|gegen)?"
                        r"\s*(?:godz\.)?\s*$")
TODAY = re.compile(r"\b(?:(?i:dziś|dzisiaj|today|heute|i dag|dagens dato)|Stand)\b")
SENT_END = re.compile(r"[.;!?](?=\s+[A-ZĄĆĘŁŃÓŚŹŻÄÖÜÆØÅ\"'(\[-])|\n")

PRESENT = r"(?:obecnie|nadal|dziś|present|now|today|current|heute|jetzt|aktuell|dato|nå|i dag)"
MONTH_RANGE = re.compile(r"(?<![\d./\-])(0?[1-9]|1[0-2])[./](\d{4})\s{0,3}(?:-|–|—|to|bis|til|do)\s{0,3}"
                         r"(?:(0?[1-9]|1[0-2])[./](\d{4})(?![\d./])|" + PRESENT + r"\b)", re.I)

WORKDAY_WORDS = re.compile(r"\b(?:dni robocz|dnia robocz|dzień robocz|working day|business day|werktag|arbeitstag|"
                           r"virkedag|arbeidsdag|yrkedag)", re.I)
HOLIDAY_WORDS = re.compile(r"\b(?:wolny|wolnym|wolne|wolnymi|świę|święt|holiday|feiertag|arbeitsfrei|helligdag|fridag|"
                           r"høytidsdag|offentlig fri)", re.I)
_DUR_UNITS = [
    ("wd", r"dni robocz\w*|dnia robocz\w*|dzień robocz\w*|working days?|business days?|werktag\w*|arbeitstag\w*|"
           r"virkedag\w*|arbeidsdag\w*|yrkedag\w*"),
    ("d", r"kalendertag\w*|kalenderdag\w*|calendar days?|dni|dnia|dzień|days?|tage[ns]?|tag|dager|dagar|dag|døgn"),
    ("w", r"tydzień|tygodni[ae]?|weeks?|wochen?|uker|uke"),
    ("m", r"miesiąc[ea]?|miesięcy|months?|monat(?:e|en|s)?|måned(?:er)?"),
    ("y", r"lat[a]?|rok|roku|years?|jahr(?:e|en|es)?|år"),
]
DURATION = re.compile(r"(?<![\w,.])(\d{1,4})\s?(?:-\s?)?(?:(?:calendar|full|clear|kalendarzow\w*|"
                      r"pełn\w*|volle[n]?|hele)\s+)?(" + "|".join(f"(?P<{u}>{rx})" for u, rx in _DUR_UNITS) + r")\b",
                      re.I)
ORDINAL_DAY = re.compile(r"(?<![\w,.])(\d{1,3})\.\s?(?:dnia|dzień|tag|tages|dag|werktag\w*|arbeitstag\w*|virkedag\w*)\b"
                         r"(\s+robocz\w*)?", re.I)
PAST = re.compile(r"^\s*(?:temu|ago|siden|zuvor|her)\b", re.I)

# numbers with a unit: currencies and common measurement units
_CUR_PREFIX = {"£": "GBP", "€": "EUR", "$": "USD", "gbp": "GBP", "eur": "EUR", "usd": "USD", "nok": "NOK", "pln": "PLN",
               "chf": "CHF", "sek": "SEK", "dkk": "DKK", "kr": "NOK", "kr.": "NOK"}
_UNIT_SUFFIX = [
    ("PLN", r"zł|PLN|złotych|zlotych"), ("EUR", r"EUR|euro|€"), ("GBP", r"GBP|£|pounds?"), ("USD", r"USD|\$|dollars?"),
    ("NOK", r"NOK|kroner|kr"), ("SEK", r"SEK"), ("DKK", r"DKK"), ("CHF", r"CHF"),
    ("%", r"%|procent\w*|percent|Prozent|prosent"),
    ("TB", r"TB"), ("GB", r"GB"), ("MB", r"MB"), ("kB", r"kB|KB"),
    ("kg", r"kg"), ("km", r"km"), ("cm", r"cm"), ("mm", r"mm"), ("m²", r"m²|m2"),
    ("h", r"h|godz\.|godzin[ay]?|godzina|hours?|hrs?|Std\.|Stunden?|timer|time"),
    ("min", r"min\.?|minut[ay]?|minutes?|Minuten?|minutter"),
    ("pt", r"pkt\.?|punkt\w*|points?|Punkte?n?|poeng"),
    ("ECTS", r"ECTS"),
]
# Groups of three digits after a space (also no-break and narrow no-break), ".", "," or an apostrophe ("1'234.50");
# ",-" is a whole amount ("1 234,-"). Bounded: no backtracking blow-up.
_NUM = (r"(?:\d{1,3}(?:[ \u00a0\u202f.,'’]\d{3}){1,6}(?:[.,]\d{1,6})?|\d{1,15}(?:[.,]\d{1,6})?)"
        r"(?:,[-–](?![\s\-–]?\d))?")
AMOUNT_SUFFIX = re.compile(r"(?<![\w.,/\-−'’])(?:(" + _NUM + r")\s?[-–]\s?)?(" + _NUM + r")\s?(?:"
                           + "|".join(f"(?P<u{i}>{rx})"
                                      for i, (u, rx) in enumerate(_UNIT_SUFFIX)) + r")(?![\w])")
_SUFFIX_GROUPS = [(f"u{i}", u) for i, (u, _) in enumerate(_UNIT_SUFFIX)]
AMOUNT_PREFIX = re.compile(r"(?<![\-−'’])(£|€|\$|\b(?:GBP|EUR|USD|NOK|PLN|CHF|SEK|DKK|kr\.?))\s?(" + _NUM + r")(?![\d])", re.I)

PERSONS = re.compile(r"\b(\d{1,2})\s+(?:osob[ya]?|osób|członk\w+|persons?|people|Personen|personer|"
                     r"members|Mitglieder|medlemmer)\b", re.I)
PER_PERSON = re.compile(r"\b(?:na osobę|na jedną osobę|per person|per capita|pro Person|pro Kopf|per person|"
                        r"per hode)\b", re.I)
MONEY = {"PLN", "EUR", "GBP", "USD", "NOK", "SEK", "DKK", "CHF"}
NOT_ADDITIVE = {"mm", "cm", "m²"}  # dimensions of one piece are compared, not added
# The rule speaks of a total: amounts of one sentence are then added up (a list is always added up).
SUM_WORDS = re.compile(r"\b(?:sum(?!mar(?:y|ies)|mers?\b)\w*|łączn\w*|razem|dodaj\w*|total\w*|combined|altogether|"
                       r"add(?!ress|ition)\w*|together|"
                       r"insgesamt|zusammen\w*|gesamt\w*|addier\w*|til sammen|samlet|totalt|legg\w* sammen)\b", re.I)
# The rule counts from today: today + / - each duration is computed.
TODAY_RULE = re.compile(r"\b(?:dziś|dzisiaj|dzisiejsz\w*|today|heute|heutig\w*|i dag|dagens)\b", re.I)
LIST_LINE = re.compile(r"^\s*(?:[-•*–]|\d{1,2}[.)])\s+")

ID_TRIGGERS = {
    "account": re.compile(r"rachun\w*|konto bankow|numer konta|IBAN|account number|bank account|bank details|"
                          r"Kontonummer|Bankverbindung|Kontoverbindung|kontonummer|bankkonto", re.I),
    "email": re.compile(r"e-?mail|e-postadress|epostadress", re.I),
    "postal": re.compile(r"kod\w* pocztow|post ?code|postal code|zip code|Postleitzahl|PLZ|postnummer", re.I),
}
ID_PATTERNS = {
    "account": re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b|"  # IBAN
                          r"(?<![\d-])\d{2}(?:[ ]?\d{4}){6}(?![\d-])|"  # Polish NRB (26 digits)
                          r"(?<![\d.-])\d{4}[ .]\d{2}[ .]\d{5}(?![\d.-])|"  # Norwegian account (11 digits)
                          r"\b\d{2}-\d{2}-\d{2}\b\D{0,20}\b\d{8}\b"),  # UK sort code and account
    "email": re.compile(r"(?<![\w.+-])[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,8}"),
    "postal": re.compile(r"(?<![\d-])\d{2}-\d{3}(?![\d-])|\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b|"
                         r"(?<![\d-])\d{4,5}\s+[A-ZÄÖÜÆØÅ][a-zäöüßæøå]+"),
}


# ------------------------------------------------------------------ helpers

def detect_language(text: str) -> str:
    """The most likely of pl / en / de / nb by frequent short words; English when nothing matches."""
    words = re.findall(r"[^\W\d_]+", text)
    lower = [w.lower() for w in words]
    scores = {}
    for lang, stop in STOPWORDS.items():
        s = {w.lower() for w in stop}
        scores[lang] = sum(w in s for w in lower)
    # the currency "zł" says nothing about the language of the text ("The fine was 12.500 zł.")
    scores["pl"] += 2 * len(re.findall(r"[ąćęłńśźż]", re.sub(r"\bzł\b", "", text), re.I))
    scores["de"] += 2 * len(re.findall(r"[äöüß]", text, re.I))
    scores["nb"] += 2 * len(re.findall(r"[æøå]", text, re.I))
    best = max(LANGUAGES, key=lambda k: scores[k])
    return best if scores[best] > 0 else "en"


def _sentence_bounds(text: str) -> list[int]:
    return [m.start() for m in SENT_END.finditer(text)]


def _sent(bounds: list[int], pos: int) -> int:
    return bisect_right(bounds, pos - 1)


def _plural(lang: str, unit: str, n: int) -> str:
    one, many = UNIT_WORDS[lang][unit]
    if lang == "pl" and unit == "y" and 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        many = "lata"
    return one if n == 1 else many


def _add_months(d: date, k: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 + k, 12)
    m += 1
    last = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)).day
    return date(y, m, min(d.day, last))


def _full_months(a: date, b: date) -> int:
    """Whole calendar months from a to b (b >= a)."""
    n = (b.year - a.year) * 12 + b.month - a.month
    if _add_months(a, n) > b:
        n -= 1
    return max(n, 0)


def _is_workday(d: date, holidays: set[date]) -> bool:
    return d.weekday() < 5 and d not in holidays


def _workdays_between(a: date, b: date, holidays: set[date]) -> int:
    """Working days from the day after a to b inclusive (whole weeks counted at once)."""
    days = (b - a).days
    if days <= 0:
        return 0
    weeks, rest = divmod(days, 7)
    n = 5 * weeks + sum((a.weekday() + i) % 7 < 5 for i in range(1, rest + 1))
    weekdays = sorted(h for h in holidays if h.weekday() < 5)  # at most MAX_HOLIDAYS
    return n - (bisect_right(weekdays, b) - bisect_right(weekdays, a))


def _plus_workdays(a: date, k: int, holidays: set[date]) -> date:
    d = a
    while k > 0:
        d += timedelta(days=1)
        k -= _is_workday(d, holidays)
    return d


def _plus(d: date, n: int, unit: str, holidays: set[date]) -> date:
    if unit == "d":
        return d + timedelta(days=n)
    if unit == "wd":
        return _plus_workdays(d, n, holidays)
    if unit == "w":
        return d + timedelta(weeks=n)
    return _add_months(d, n * (12 if unit == "y" else 1))


# ------------------------------------------------------------------ dates

class _Date:
    __slots__ = ("start", "end", "day", "time")

    def __init__(self, start, end, day, time=None):
        self.start, self.end, self.day, self.time = start, end, day, time


def _make_date(y, m, d):
    try:
        if not 1900 <= y <= 2200:
            return None
        return date(y, m, d)
    except ValueError:
        return None


def find_dates(text: str) -> list[_Date]:
    """Full dates (with a year) in order of appearance. Numeric d/m/y dates are read day first (European order),
    month first when the day-first reading is impossible; all slash dates are read month first when the text has a
    slash date that can only be month first (04/25/2027) and none that can only be day first."""
    found = []
    for m in islice(DATE_TXT.finditer(text), MAX_MATCHES):
        d = _make_date(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
        if d:
            found.append(_Date(m.start(), m.end(), d))
    for m in islice(DATE_US.finditer(text), MAX_MATCHES):
        d = _make_date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
        if d:
            found.append(_Date(m.start(), m.end(), d))
    for m in islice(DATE_ISO.finditer(text), MAX_MATCHES):
        d = _make_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            found.append(_Date(m.start(), m.end(), d))
    numeric = list(islice(DATE_DMY.finditer(text), MAX_MATCHES))
    # A text that writes 04/25/2027 writes its other slash dates month first too, so they are read the same way.
    slashes = [(int(m.group(1)), int(m.group(3))) for m in numeric if m.group(2) == "/"]
    month_first = any(b > 12 >= a for a, b in slashes) and not any(a > 12 >= b for a, b in slashes)
    for m in numeric:
        a, b, y = int(m.group(1)), int(m.group(3)), int(m.group(4))
        if month_first and m.group(2) == "/":
            a, b = b, a
        d = _make_date(y, b, a) or _make_date(y, a, b)
        if d:
            found.append(_Date(m.start(), m.end(), d))
    found.sort(key=lambda x: (x.start, -x.end))
    out, last_end = [], -1
    for x in found:  # drop overlapping matches (a US-style match inside a day-first one)
        if x.start >= last_end:
            out.append(x)
            last_end = x.end
    out = out[:MAX_MATCHES]
    times = [(m.start(), m.end(), int(m.group(1)), int(m.group(2)))
             for m in islice(TIME.finditer(text), MAX_MATCHES)]
    starts = [t[0] for t in times]
    for i, x in enumerate(out):
        lo = out[i - 1].end if i else 0
        hi = out[i + 1].start if i + 1 < len(out) else len(text)
        # only the times within 30 characters of the date (bisect over the sorted time positions)
        near = times[bisect_left(starts, max(lo, x.start - 30)):bisect_left(starts, min(hi, x.end + 20))]
        after = [t for t in near if x.end <= t[0] and TIME_AFTER.match(text[x.end:t[0]])]
        before = [t for t in near if t[1] <= x.start and TIME_BEFORE.match(text[t[1]:x.start])]
        t = before[-1] if before else after[0] if after else None
        if t:
            x.time = (t[2], t[3])
    return out


def _holidays(texts: Sequence[str], years: Sequence[int]) -> set[date]:
    """Dates the text names as days off: dates in a sentence that speaks of a holiday / day off. One budget of
    MAX_MATCHES dates for all sentences, at most MAX_HOLIDAYS days off; a year-less date ("1 May") is taken in each of
    `years`."""
    out: dict[date, None] = {}  # ordered, without duplicates
    budget = MAX_MATCHES
    for text in texts:
        bounds = [0] + [b + 1 for b in _sentence_bounds(text)] + [len(text)]
        for a, b in zip(bounds, bounds[1:]):
            if budget <= 0 or len(out) >= MAX_HOLIDAYS:
                break
            s = text[a:b]
            if not HOLIDAY_WORDS.search(s):
                continue
            full = find_dates(s)[:budget]
            budget -= len(full)
            out.update((x.day, None) for x in full)
            for m in islice(DATE_NO_YEAR.finditer(s), max(budget, 0)):
                budget -= 1
                for y in years:
                    d = _make_date(y, MONTHS[m.group(2).lower()], int(m.group(1)))
                    if d:
                        out[d] = None
    return set(list(out)[:MAX_HOLIDAYS])


def _today(text: str, dates: list[_Date]) -> _Date | None:
    for m in islice(TODAY.finditer(text), MAX_MATCHES):
        for x in dates:
            if m.end() <= x.start <= m.end() + 25 and "\n" not in text[m.end():x.start]:
                return x
    return None


def _durations(texts: Sequence[str]) -> list[tuple[int, str]]:
    out = []
    for text in texts:
        for m in islice(DURATION.finditer(text), MAX_MATCHES):
            unit = next(u for u, _ in _DUR_UNITS if m.group(u))
            n = int(m.group(1))
            if n == 0 or PAST.match(text[m.end():]) or text[max(0, m.start() - 4):m.start()].lower() == "vor ":
                continue
            if unit in ("d", "wd") and n > 3650 or unit in ("w", "m") and n > 240 or unit == "y" and n > 30:
                continue
            if (n, unit) not in out:
                out.append((n, unit))
        for m in islice(ORDINAL_DAY.finditer(text), MAX_MATCHES):
            n = int(m.group(1))
            work = bool(m.group(2)) or bool(re.search(r"robocz|werk|arbeit|virke|working|business", m.group(0), re.I))
            unit = "wd" if work else "d"
            if n and (n, unit) not in out:
                out.append((n, unit))
    return out


# ------------------------------------------------------------------ numbers

def _number(raw: str, lang: str, money: bool = False) -> Decimal | None:
    """The value of a number as written, or None when it is malformed or ambiguous. With both "." and "," the last
    one is the decimal separator ("1.234,56", "1,234.56"); ",-" means no decimals, so dots before it group thousands
    ("1.234,-"). With one kind of separator followed by groups of exactly three digits, the language of the text
    decides: the separator that is not its decimal separator groups thousands (English "12,500" = 12500; Polish,
    German, Norwegian "12.500" = 12500). Its decimal separator before exactly three digits is a decimal for a
    measurement ("25,055 mm" in Polish); for money ("12,500 zł" in Polish, "£12.500" in English) it can mean 12.5 or
    12500, so the amount is not read."""
    s = re.sub(r"[\u00a0\u202f'’]", " ", raw)
    whole = s.endswith((",-", ",–"))
    if whole:
        s = s[:-2]
    if re.fullmatch(r"\d{1,3}(?: \d{3})+(?:[.,]\d+)?", s):
        s = s.replace(" ", "")
    if " " in s:
        return None
    dots, commas = s.count("."), s.count(",")
    if dots and commas:  # the last separator is the decimal one
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        group = "." if dec == "," else ","
        if not re.fullmatch(rf"[1-9]\d{{0,2}}(?:{re.escape(group)}\d{{3}})+{re.escape(dec)}\d+", s):
            return None
        s = s.replace(group, "").replace(dec, ".")
    elif dots or commas:
        sep = "." if dots else ","
        grouped = re.fullmatch(rf"[1-9]\d{{0,2}}(?:{re.escape(sep)}\d{{3}})+", s)  # "0,500" is never grouped
        if whole or (grouped and (dots + commas > 1 or sep != ("." if lang == "en" else ","))):
            if not grouped:
                return None
            s = s.replace(sep, "")  # thousands
        elif grouped and money:
            return None  # "12,500 zł" in Polish, "£12.500" in English: 12.5 or 12500
        elif dots + commas > 1:
            return None  # "1.2.3"
        else:
            s = s.replace(sep, ".")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _fmt_number(x: Decimal, lang: str, places: int) -> str:
    q = x.quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP) if places else x.quantize(Decimal(1), ROUND_HALF_UP)
    s = f"{q:,.{places}f}"
    if lang == "en":
        return s
    sep = "." if lang == "de" else " "
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", sep)


class _Amount:
    __slots__ = ("start", "end", "value", "unit", "places", "prefix", "symbol", "bound")

    def __init__(self, start, end, value, unit, places, prefix, symbol, bound=False):
        self.start, self.end, self.value, self.unit = start, end, value, unit
        self.places, self.prefix, self.symbol = places, prefix, symbol
        self.bound = bound  # an end of a range ("24,95-25,05 mm") or a tolerance ("± 0,05 mm"): never summed


def _places(raw: str, lang: str) -> int:
    """Decimal places as written (0 for "1 234,-" and when the last separator groups thousands)."""
    m = re.search(r"[.,](\d+)$", raw)
    if not m:
        return 0
    sep = raw[m.start()]
    if len(m.group(1)) == 3 and ("." if sep == "," else ",") not in raw and (
            raw.count(sep) > 1 or sep != ("." if lang == "en" else ",")):
        return 0  # a thousands separator
    return len(m.group(1))


def find_amounts(text: str, lang: str, masked: Sequence[tuple[int, int]] = ()) -> list[_Amount]:
    """Numbers with a unit (currency, %, data size, weight, length, hours, ...), in order of appearance. A range
    "24,95-25,05 mm" gives two amounts. Spans in `masked` (dates) are skipped."""
    out = []
    masked = sorted(masked)
    mstarts = [x for x, _ in masked]

    def overlaps(spans, starts, a, b):
        """Whether [a, b) overlaps one of the sorted, non-overlapping spans."""
        i = bisect_left(starts, b)
        return i > 0 and spans[i - 1][1] > a

    def free(a, b):
        return not overlaps(masked, mstarts, a, b)

    for m in islice(AMOUNT_PREFIX.finditer(text), MAX_MATCHES):
        if not free(m.start(), m.end()):
            continue
        v = _number(m.group(2), lang, money=True)
        if v is not None:
            sym = m.group(1)
            out.append(_Amount(m.start(), m.end(), v, _CUR_PREFIX[sym.lower()], _places(m.group(2), lang), True, sym))
    taken = sorted((a.start, a.end) for a in out)
    tstarts = [x for x, _ in taken]
    for m in islice(AMOUNT_SUFFIX.finditer(text), MAX_MATCHES):
        if not free(m.start(), m.end()) or overlaps(taken, tstarts, m.start(), m.end()):
            continue
        unit = next(u for g, u in _SUFFIX_GROUPS if m.group(g))
        sym = next(m.group(g) for g, u in _SUFFIX_GROUPS if m.group(g))
        if unit == "h" and sym == "time" and lang != "nb":
            continue
        if unit == "NOK" and sym.lower().startswith("kr") and lang == "pl":
            continue
        ranged = m.group(1) is not None
        if text[max(0, m.start() - 3):m.start()].rstrip().endswith(("±", "+/-")):
            continue  # a tolerance
        for raw, start in ((m.group(1), m.start(1)), (m.group(2), m.start(2))):
            if raw is None:
                continue
            v = _number(raw, lang, money=unit in MONEY)
            if v is not None:
                out.append(_Amount(start, m.end(), v, unit, _places(raw, lang), False, sym, ranged))
    out.sort(key=lambda a: a.start)
    return out[:MAX_MATCHES]


def _render_amount(a_value: Decimal, like: _Amount, lang: str, places: int | None = None) -> str:
    places = like.places if places is None else places
    num = _fmt_number(a_value, lang, places)
    sym = like.symbol
    if like.prefix:
        return f"{sym}{num}" if sym in "£€$" else f"{sym} {num}"
    if sym == "%":
        return f"{num}%"
    return f"{num} {sym}"


def _strip_numbers(line: str) -> str:
    s = re.sub(r"[£€$]?\s?\d[\d  .,:/\-]*\s?(?:" + "|".join(rx for _, rx in _UNIT_SUFFIX) + r")?", " ", line)
    s = re.sub(r"[-•*–:;,()]", " ", s)
    return " ".join(s.lower().split())


# ------------------------------------------------------------------ facts

def facts(state: str, instructions: Sequence[str] = (), language: str | None = None) -> tuple[list[str], str]:
    """The facts for a state and the instructions of its questions, as (lines, language). Empty lines when there is
    nothing to compute. Dates and amounts come from the state; durations, thresholds and named days off also from the
    instructions."""
    state = state[:MAX_INPUT_CHARS]
    rule = "\n".join(instructions)[:MAX_INPUT_CHARS]
    lang = language or detect_language(state + "\n" + rule)
    if lang not in LANGUAGES:
        lang = "en"
    t, wd = T[lang], WEEKDAYS[lang]
    lines: list[str] = []

    # --- dates
    dates = find_dates(state)
    years = list(dict.fromkeys(x.day.year for x in dates + find_dates(rule)))[:MAX_YEARS]
    holidays = _holidays([state, rule], years)
    workdays = bool(WORKDAY_WORDS.search(state) or WORKDAY_WORDS.search(rule))
    today = _today(state, dates)
    seen, events = set(), []
    for x in dates:
        if x.day in seen:
            continue
        seen.add(x.day)
        if x.day in holidays and (today is None or x.day != today.day):
            continue
        events.append(x)
    events = events[:MAX_DATES]
    def day(d):
        return wd[d.weekday()] + (f"; {t['holiday']}" if d in holidays else "")

    def counts(lo, hi):
        """The working days (when the text speaks of them) and the full months / years from lo to hi."""
        out = ""
        if workdays:
            out += "; " + t["workdays"].format(w=_workdays_between(lo, hi, holidays), a=lo.isoformat(),
                                               b=hi.isoformat(), hol=t["hol"] if holidays else "")
        span = _span(lo, hi, t)
        return out + ("; " + span if span else "")

    for x in events:
        line = f"{x.day.isoformat()} = {day(x.day)}"
        if today is not None:
            n = (x.day - today.day).days
            if n == 0:
                line += f" = {t['today']}"
            else:
                line += "; " + (t["before"] if n < 0 else t["after"]).format(n=abs(n), u=_plural(lang, "d", abs(n)))
                line += counts(*sorted((x.day, today.day)))
        lines.append(line)
    others = [x for x in events if today is None or x.day != today.day]
    pairs = (list(zip(others, others[1:])) if len(others) > 4 or today is not None and len(others) > 3
             else [(a, b) for i, a in enumerate(others) for b in others[i + 1:]])
    for a, b in pairs[:MAX_GAPS if today is None else 3]:
        lo, hi = sorted((a.day, b.day))
        n = (hi - lo).days
        lines.append(t["gap"].format(a=lo.isoformat(), b=hi.isoformat(), n=n, u=_plural(lang, "d", n), n1=n + 1)
                     + counts(lo, hi))

    # --- clock time between dated times
    timed = [x for x in dates if x.time]
    for a, b in list(zip(timed, timed[1:]))[:MAX_HOURS]:
        da, db = datetime(a.day.year, a.day.month, a.day.day, *a.time), datetime(b.day.year, b.day.month, b.day.day, *b.time)
        lo, hi = sorted((da, db))
        mins = int((hi - lo).total_seconds() // 60)
        lines.append(t["hours"].format(a=lo.strftime("%Y-%m-%d %H:%M"), b=hi.strftime("%Y-%m-%d %H:%M"),
                                       h=mins // 60, m=mins % 60))

    # --- date + duration
    durs = _durations([rule, state])[:MAX_DURATIONS]
    anchors = others[:MAX_ANCHORS]
    plus = []
    for x in anchors:
        for n, unit in durs:
            d = _plus(x.day, n, unit, holidays)
            plus.append(f"{x.day.isoformat()} + {n} {_plural(lang, unit, n)} = {d.isoformat()} ({day(d)})")
    if today is not None and TODAY_RULE.search(rule):
        tp = []
        for n, unit in durs:
            d = _plus(today.day, n, unit, holidays)
            tp.append(f"{today.day.isoformat()} ({t['today']}) + {n} {_plural(lang, unit, n)} = {d.isoformat()} ({day(d)})")
            if unit != "wd":
                back = _plus(today.day, -n, unit, holidays)
                tp.append(f"{today.day.isoformat()} ({t['today']}) - {n} {_plural(lang, unit, n)} = {back.isoformat()} "
                          f"({day(back)})")
        lines += tp[:MAX_TODAY_PLUS]
    lines += plus[:MAX_PLUS]
    lines = lines[:MAX_CALENDAR]

    # --- month ranges ("09.2019 - 08.2021", "03.2020 - obecnie")
    ref = today.day if today is not None else None
    ranges = []
    for m in islice(MONTH_RANGE.finditer(state), MAX_MATCHES):
        a = (int(m.group(2)), int(m.group(1)))
        if m.group(3):
            b = (int(m.group(4)), int(m.group(3)))
        elif ref is not None:
            b = (ref.year, ref.month)
        else:
            continue
        if b < a:
            continue
        n = (b[0] - a[0]) * 12 + b[1] - a[1] + 1
        ranges.append((a, b))
        lines.append(t["mrange"].format(a=f"{a[1]:02d}.{a[0]}", b=f"{b[1]:02d}.{b[0]}", n=n))
        if len(ranges) >= MAX_MONTH_RANGES:
            break
    if len(ranges) > 1:
        months = set()
        for a, b in ranges:
            k = a[0] * 12 + a[1]
            while k <= b[0] * 12 + b[1]:
                months.add(k)
                k += 1
        lines.append(t["mtotal"].format(n=len(months), y=len(months) // 12, m=len(months) % 12))

    # --- amounts
    masked = [(x.start, x.end) for x in dates]
    amounts = find_amounts(state, lang, masked)
    thresholds = find_amounts(rule, lang, [(x.start, x.end) for x in find_dates(rule)])
    sums = _sums(state, amounts, lang, sentences=bool(SUM_WORDS.search(rule)))
    lines += [s for s, _, _ in sums][:MAX_SUMS]
    computed = [(v, like, max(like.places, 0)) for _, v, like in sums[:MAX_SUMS]]
    if PER_PERSON.search(rule) or PER_PERSON.search(state):
        persons = {int(m.group(1)) for m in PERSONS.finditer(state)}
        if len(persons) == 1:
            n = persons.pop()
            if n > 1:
                money = [a for a in amounts if a.unit in MONEY]
                for a in money[:MAX_PER_PERSON]:
                    r = a.value / n
                    lines.append(t["per_person"].format(a=_render_amount(a.value, a, lang), n=n,
                                                        r=_render_amount(r, a, lang, 2)))
                    computed.append((r, a, 2))
    lines += _comparisons(amounts, computed, thresholds, lang)

    # --- identifiers asked for by the instructions
    ids = []
    for kind, trigger in ID_TRIGGERS.items():
        if not trigger.search(rule):
            continue
        m = ID_PATTERNS[kind].search(state)
        what = t["what"][kind]
        ids.append(t["found"].format(what=what, value=m.group(0).strip()) if m else t["missing"].format(what=what))
    lines += ids[:MAX_IDS]
    return lines, lang


def _span(lo: date, hi: date, t: dict) -> str:
    m = _full_months(lo, hi)
    if m < 1:
        return ""
    if m < 12:
        return t["months"].format(n=m)
    return t["span"].format(y=m // 12, m=m % 12)


def _sums(state: str, amounts: list[_Amount], lang: str, sentences: bool = True) -> list[tuple[str, Decimal, _Amount]]:
    """Sums of same-unit amounts: of one sentence, of the lines of a list, and of list lines that share a
    description (sentences only when `sentences`). Returns (line, value, a representative amount)."""
    out, done = [], set()

    def add(group):
        group = [a for a in group if not a.bound and a.unit not in NOT_ADDITIVE]
        key = tuple(id(a) for a in group)
        vals = tuple((a.unit, a.value) for a in group)
        if len(group) < 2 or len(group) > 8 or key in done or vals in done:
            return
        done.add(key)
        done.add(vals)
        places = max(a.places for a in group)
        total = sum((a.value for a in group), Decimal(0))
        out.append((" + ".join(_render_amount(a.value, a, lang, places) for a in group) + " = "
                    + _render_amount(total, group[0], lang, places), total, group[0]))

    # list lines
    lines, pos = [], 0
    for raw in state.split("\n"):
        lines.append((pos, pos + len(raw), raw))
        pos += len(raw) + 1
    starts = [x.start for x in amounts]  # amounts are sorted by position
    blocks, cur = [], []
    for a, b, raw in lines:
        if LIST_LINE.match(raw):
            cur.append((a, b, raw))
        else:
            if len(cur) > 1:
                blocks.append(cur)
            cur = []
    if len(cur) > 1:
        blocks.append(cur)
    for block in blocks:
        by_unit: dict[str, list[tuple[_Amount, str]]] = {}
        for a, b, raw in block:
            here = amounts[bisect_left(starts, a):bisect_left(starts, b)]
            if len(here) == 1:  # one amount per line: a list of amounts
                by_unit.setdefault(here[0].unit, []).append((here[0], _strip_numbers(raw)))
        for unit, items in by_unit.items():
            labels: dict[str, list[_Amount]] = {}
            for x, label in items:
                labels.setdefault(label, []).append(x)
            if len(labels) > 1:
                for group in labels.values():
                    add(group)
            add([x for x, _ in items])
    # sentences (and lines that are not list items)
    bounds = _sentence_bounds(state)
    by_sent: dict[tuple[int, str], list[_Amount]] = {}
    for x in amounts:
        line_start = state.rfind("\n", 0, x.start) + 1
        if LIST_LINE.match(state[line_start:x.start + 1]):
            continue
        by_sent.setdefault((_sent(bounds, x.start), x.unit), []).append(x)
    if sentences:
        for group in by_sent.values():
            add(group)
    return out


def _comparisons(amounts, computed, thresholds, lang) -> list[str]:
    """Amounts of the state (and computed sums) against the thresholds of the same unit in the instructions, as a
    symbol. When there are more pairs than MAX_COMPARE, the pairs whose two values are closest are kept: an amount
    far from a threshold is clearly on one side of it."""
    th: list[_Amount] = []
    for x in thresholds:
        if all((x.unit, x.value) != (y.unit, y.value) for y in th):
            th.append(x)
    subjects: list[tuple[Decimal, _Amount, int]] = list(computed)
    for a in amounts:
        if all((a.unit, a.value) != (b.unit, v) for v, b, _ in subjects):
            subjects.append((a.value, a, a.places))
    pairs = []
    for i, (v, a, places) in enumerate(subjects):
        for j, x in enumerate(th):
            if x.unit != a.unit:
                continue
            lo, hi = sorted((abs(v), abs(x.value)))
            closeness = float(hi / lo) if lo else math.inf
            pairs.append((closeness, i, j))
    keep = sorted(sorted(pairs)[:MAX_COMPARE], key=lambda p: (p[1], p[2]))
    out = []
    for _, i, j in keep:
        v, a, places = subjects[i]
        x = th[j]
        op = ">" if v > x.value else "<" if v < x.value else "="
        out.append(f"{_render_amount(v, a, lang, max(places, x.places if v != v.to_integral_value() else 0))} {op} "
                   f"{_render_amount(x.value, x, lang)}")
    return out


def inject(state: str, lines: Sequence[str], language: str) -> str:
    """The state with the facts block appended (unchanged when there are no facts)."""
    if not lines:
        return state
    return state + "\n\n" + HEADER.get(language, HEADER["en"]) + "\n" + "\n".join(f"- {x}" for x in lines)
