"""`"facts": true`: calendar and number arithmetic computed in code and appended to the state. Only arithmetic and
calendar facts, never a verdict; the default request is unchanged."""

import math
from datetime import date

import pytest

from bielik_one.api import server
from bielik_one.api.server import LetterEngine, SystemOneIn
from bielik_one.prompting.facts import HEADER, detect_language, facts, find_dates, inject


def lines_of(state, *instructions, language=None):
    return facts(state, list(instructions), language)[0]


# ------------------------------------------------------------------ language


@pytest.mark.parametrize(
    "text, language",
    [
        ("Klient złożył reklamację w terminie i prosi o zwrot pieniędzy na rachunek bankowy.", "pl"),
        ("The customer filed the complaint on time and asks for a refund to the bank account.", "en"),
        ("Der Kunde hat die Reklamation fristgerecht eingereicht und bittet um die Rückzahlung.", "de"),
        ("Kunden har levert reklamasjonen i tide og ber om at pengene blir betalt tilbake.", "nb"),
    ],
)
def test_language_detection(text, language):
    assert detect_language(text) == language


def test_unknown_language_falls_back_to_english():
    assert detect_language("12345 67890") == "en"


# ------------------------------------------------------------------ dates


@pytest.mark.parametrize(
    "text, expected",
    [
        ("dnia 3 marca 2025 r.", date(2025, 3, 3)),
        ("on 1 February 2027", date(2027, 2, 1)),
        ("on February 1, 2027", date(2027, 2, 1)),
        ("am 4. Mai 2027", date(2027, 5, 4)),
        ("den 4. mai 2027", date(2027, 5, 4)),
        ("2026-11-12", date(2026, 11, 12)),
        ("12.11.2026", date(2026, 11, 12)),
        ("12/04/2027", date(2027, 4, 12)),  # day first
        ("04/25/2027", date(2027, 4, 25)),  # month first only when day first is impossible
    ],
)
def test_date_formats(text, expected):
    assert [x.day for x in find_dates(text)] == [expected]


def test_invalid_dates_and_version_numbers_are_not_dates():
    assert find_dates("31.02.2026, wersja 1.2.3, numer 2026-13-45") == []


def test_today_relation_weekday_and_deadline():
    lines = lines_of(
        "Today is Wednesday 10 March 2027. The decision was delivered on 15 February 2027.",
        "An appeal is possible within 21 days of delivery.",
    )
    assert "2027-03-10 = Wednesday = today" in lines
    assert "2027-02-15 = Monday; 23 days before today" in lines
    assert "2027-02-15 + 21 days = 2027-03-08 (Monday)" in lines
    assert not any("(today) " in x for x in lines)  # the rule does not count from today


def test_today_plus_and_minus_when_the_rule_counts_from_today():
    lines = lines_of(
        "Today is Wednesday 10 March 2027. Requested expiry: 12 April 2027.",
        "Rights may expire no later than 30 days after today.",
    )
    assert "2027-03-10 (today) + 30 days = 2027-04-09 (Friday)" in lines
    assert "2027-03-10 (today) - 30 days = 2027-02-08 (Monday)" in lines


def test_gap_counts_both_ways_and_singular():
    lines = lines_of("Der Antrag ging am 2026-06-01 ein und wurde am 2026-06-02 beantwortet.", "War die Antwort rechtzeitig?")
    assert "Vom 2026-06-01 bis 2026-06-02: 1 Tag (2 mit beiden Tagen)" in lines


def test_working_days_skip_weekends_and_the_days_off_the_text_names():
    lines = lines_of(
        "Dostawę przyjęto w czwartek 2026-04-30. Reklamację wysłano 2026-05-08.",
        "Reklamację składa się w ciągu 5 dni roboczych od przyjęcia dostawy. Dni wolne od pracy: 1 maja i 3 maja.",
    )
    # 1 May (Fri) is a day off named in the text, 2-3 May a weekend: 4, 5, 6, 7, 8 May are the working days.
    assert any(x.startswith("Od 2026-04-30 do 2026-05-08: 8 dni") and x.endswith(": 5") for x in lines)
    assert "2026-04-30 + 5 dni roboczych = 2026-05-08 (piątek)" in lines


def test_a_named_day_off_is_marked():
    lines = lines_of(
        "Vedtaket ble levert 2026-04-17.",
        "Klagefristen er 14 dager. 2026-05-01 er en offentlig fridag.",
    )
    assert "2026-04-17 + 14 dager = 2026-05-01 (fredag; fridag nevnt i teksten)" in lines


def test_ordinal_day_of_a_deadline():
    lines = lines_of("Przesyłkę doręczono 2026-09-01.", "Spór można zgłosić najpóźniej 14. dnia od doręczenia.")
    assert "2026-09-01 + 14 dni = 2026-09-15 (wtorek)" in lines


def test_past_durations_are_not_deadlines():
    assert not any("+ 3" in x for x in lines_of("Bestilt 2026-01-10, for 3 dager siden.", "Er det for sent?"))


def test_long_spans_in_full_years_and_months():
    lines = lines_of("Today is 20 October 2026. Date of birth: 10 March 2008.", "Is the patient under 18?")
    assert "2008-03-10 = Monday; 6798 days before today; full years: 18 (+ full months: 7)" in lines


def test_clock_time_between_dated_times():
    lines = lines_of(
        "Die Störung begann um 22:30 am Montag, 8. Februar 2027 und endete am 9. Februar 2027 um 03:05.",
        "Dauerte die Störung länger als 4 Stunden?",
    )
    assert "Von 2027-02-08 22:30 bis 2027-02-09 03:05: 4 Std. 35 Min." in lines


def test_month_ranges_and_their_total_without_overlap():
    lines = lines_of(
        "Today is 5 October 2026. Experience: accountant 09.2019 - 08.2021; senior accountant 06.2021 - present.",
        "Does the candidate have at least 5 years of experience?",
    )
    assert "09.2019 - 08.2021: 24 months (counting both months)" in lines
    assert "06.2021 - 10.2026: 65 months (counting both months)" in lines
    assert "Periods in total (overlaps counted once): 86 months = 7 years and 2 months" in lines


# ------------------------------------------------------------------ amounts


def test_comparison_with_the_threshold_of_the_rule_is_a_symbol():
    lines = lines_of("Kosztorys naprawy: 512,40 zł.", "Szkody o wartości nie wyższej niż 500 zł nie są wypłacane.")
    assert lines == ["512,40 zł > 500 zł"]


def test_list_lines_with_the_same_description_are_summed():
    lines = lines_of(
        "Account activity:\n- 03/01/2027 cash deposit £4,000\n- 09/01/2027 cash deposit £3,500\n"
        "- 11/01/2027 card payment £120\n- 15/01/2027 cash deposit £2,750",
        "Report if cash deposits total more than £10,000.",
    )
    assert "£4,000 + £3,500 + £2,750 = £10,250" in lines
    assert "£10,250 > £10,000" in lines


def test_amounts_in_one_sentence_are_summed_when_the_rule_speaks_of_a_total():
    lines = lines_of("Opplasting: en fil på 40 MB og en fil på 25 MB.", "Blokker hvis samlet størrelse er over 64 MB.")
    assert "40 MB + 25 MB = 65 MB" in lines
    assert "65 MB > 64 MB" in lines
    # two prices in one sentence are not a total
    assert lines_of("Elsewhere it costs £1,890, here £2,090.", "Is this a complaint about a defect?") == []


def test_number_formats_per_language():
    assert lines_of("Betrag: 12.300,50 €.", "Die Grenze liegt bei 10.000 €.") == ["12.300,50 € > 10.000 €"]
    assert lines_of("Amount: €12,300.50.", "The limit is €10,000.") == ["€12,300.50 > €10,000"]
    assert lines_of("Beløp: 12 300,50 kr.", "Grensen er 10 000 kr.") == ["12 300,50 kr > 10 000 kr"]


def test_ranges_and_tolerances_are_compared_not_added():
    lines = lines_of(
        "Nominal 25,00 ± 0,05 mm. Messung: 24,97-25,03 mm, Teil 6 = 24,94 mm.",
        "Ein Teil ist fehlerhaft, wenn der Durchmesser außerhalb von 24,95-25,05 mm liegt.",
    )
    assert "24,94 mm < 24,95 mm" in lines
    assert not any("+" in x for x in lines)
    assert not any(x.startswith("0,05") for x in lines)


def test_amount_per_person():
    lines = lines_of(
        "Household of 4 people, total net income £5,800 per month.",
        "Eligible if the monthly net income per person is no more than £1,500.",
    )
    assert "£5,800 ÷ 4 persons = £1,450.00" in lines
    assert "£1,450.00 < £1,500" in lines


def test_comparisons_keep_the_closest_pairs():
    state = "Payments: " + ", ".join(f"£{n},000" for n in (1, 2, 3, 4, 5, 6, 7, 8, 9)) + "."
    lines = lines_of(state, "Flag a payment of £9,500 or more.")
    comparisons = [x for x in lines if " < " in x or " > " in x]
    assert len(comparisons) == 6
    assert "£9,000 < £9,500" in comparisons and "£1,000 < £9,500" not in comparisons


# ------------------------------------------------------------------ identifiers


def test_identifiers_the_rule_asks_for():
    rule = "Wniosek jest kompletny, jeśli zawiera numer zamówienia i numer rachunku bankowego do zwrotu."
    assert lines_of("Zamówienie 1234. Proszę o zwrot pieniędzy.", rule) == ["W tekście nie ma: numer rachunku bankowego"]
    assert lines_of("Zamówienie 1234. Zwrot na 12 1020 3040 0000 1234 5678 9012.", rule) == [
        "W tekście jest numer rachunku bankowego: 12 1020 3040 0000 1234 5678 9012"]
    # not asked for: nothing is reported
    assert lines_of("Zamówienie 1234. Proszę o zwrot pieniędzy.", "Czy klient żąda zwrotu?") == []


# ------------------------------------------------------------------ the block


def test_no_verdicts_only_arithmetic():
    lines = lines_of(
        "Today is 1 March 2027. Invoice of £12,500 due on 15 February 2027.",
        "Escalate invoices over £10,000 that are more than 10 days overdue.",
    )
    text = " ".join(lines).lower()
    for word in ("overdue", "exceed", "late", "eligible", "within", "valid", "in time"):
        assert word not in text


def test_inject_appends_a_header_and_keeps_the_state():
    state = "Kosztorys naprawy: 512,40 zł."
    out = inject(state, ["512,40 zł > 500 zł"], "pl")
    assert out == state + "\n\n" + HEADER["pl"] + "\n- 512,40 zł > 500 zł"
    assert inject(state, [], "pl") == state


def test_nothing_to_compute():
    assert lines_of("The customer is unhappy with the colour.", "Is this a complaint?") == []


def test_explicit_language():
    assert lines_of("Amount: 600 zł.", "Limit 500 zł.", language="de") == ["600 zł > 500 zł"]
    assert facts("2026-03-02", [], "de")[0] == ["2026-03-02 = Montag"]


# ------------------------------------------------------------------ API


class StateEngine(LetterEngine):
    model_name = "decision"
    capacity = 255
    labels_threshold = 0.5
    semantic = True
    parking = None
    tok = None
    temperatures = {}

    def distributions(self, prompts, **kwargs):
        self.states = [row.state for row in prompts]
        return [{20 + i: math.log(0.5) for i, _ in enumerate(row.options)} for row in prompts], 1


@pytest.fixture(autouse=True)
def positional_tokens(monkeypatch):
    monkeypatch.setattr(server, "answer_token_ids", lambda tok, row, semantic: [20 + i for i in range(len(row.options))])


def request(**options):
    return SystemOneIn(
        model="decision",
        state=options.pop("state", "Kosztorys naprawy: 512,40 zł."),
        questions={"q": {"type": "noul", "instructions": "Czy szkoda przekracza franszyzę 500 zł?",
                         "criteria": {"true": "Tak", "false": "Nie"}}},
        **options,
    )


def test_facts_are_off_by_default():
    engine = StateEngine()
    result = engine.evaluate(request())
    assert engine.states == ["Kosztorys naprawy: 512,40 zł."]
    assert "facts" not in result


def test_facts_extend_the_state_the_model_reads():
    engine = StateEngine()
    result = engine.evaluate(request(facts=True))
    assert engine.states == ["Kosztorys naprawy: 512,40 zł.\n\n" + HEADER["pl"] + "\n- 512,40 zł > 500 zł"]
    assert result["facts"] == {"language": "pl", "lines": ["512,40 zł > 500 zł"]}
    assert set(result["answers"]) == {"q"}


def test_facts_on_a_json_state_and_without_facts():
    engine = StateEngine()
    result = engine.evaluate(request(facts=True, state={"kosztorys": "512,40 zł"}))
    assert engine.states[0].startswith('{"kosztorys": "512,40 zł"}\n\n')
    engine.evaluate(request(facts=True, state="Klient pyta o kolor."))
    assert engine.states == ["Klient pyta o kolor."]


def test_facts_flag_is_validated():
    with pytest.raises(Exception):
        request(facts="sometimes")


# ------------------------------------------------------------------ untrusted input


@pytest.mark.parametrize(
    "state, rule",
    [
        ("a.a." * 25_000, "Does the text contain an e-mail address?"),  # no "@": the e-mail pattern must not backtrack
        ("x" * 50_000 + "@" + "a." * 25_000, "Podaj adres e-mail."),
        (" ".join(f"{i % 1000:03d}" for i in range(30_000)) + " zł", "Limit 500 zł, suma łącznie."),
        ("\n".join(f"- 2026-01-{i % 28 + 1:02d} 10:{i % 60:02d} item {i} £{i},000" for i in range(5_000)),
         "Today: report if the total is more than £10,000 within 30 working days."),
        ("Today is 1 March 2027. " + "12:00 1 March 2027 " * 5_000, "Within 14 days after today."),
        ("Dziś 2026-01-01. " + "01.1900 - obecnie; " * 5_000, "Co najmniej 3 lata, na osobę, 4 osoby."),
    ],
)
def test_large_adversarial_inputs_finish_fast(state, rule):
    import time

    start = time.perf_counter()
    lines, _ = facts(state, [rule])
    assert time.perf_counter() - start < 1.0
    assert len(lines) <= 40


def test_far_apart_dates_are_counted_without_a_day_by_day_loop():
    lines = lines_of("Od 1900-01-01 do 2200-12-31.", "Ile dni roboczych upłynęło?")
    assert any(x.startswith("Od 1900-01-01 do 2200-12-31: 109937 dni") and "włącznie: 78527;" in x for x in lines)


def test_working_day_count_matches_a_day_by_day_count():
    import random
    from datetime import timedelta

    from bielik_one.prompting.facts import _workdays_between

    rng = random.Random(0)
    for _ in range(300):
        a = date(2026, 1, 1) + timedelta(days=rng.randrange(400))
        b = a + timedelta(days=rng.randrange(60))
        holidays = {a + timedelta(days=rng.randrange(-5, 70)) for _ in range(3)}
        slow = sum((a + timedelta(days=i)).weekday() < 5 and a + timedelta(days=i) not in holidays
                   for i in range(1, (b - a).days + 1))
        assert _workdays_between(a, b, holidays) == slow


# ------------------------------------------------------------------ facts and read twice


def _one_token_per_char(reads):
    """A synthetic tokenizer for the remote engine: one token per character of the state, the state twice when the
    prompt repeats it. `reads` records (state, read_twice) of every prompt built."""

    def inference_ids(tok, row, semantic=False, **kwargs):
        reads.append((row.state, kwargs.get("read_twice", False)))
        return [1] * (len(row.state) * (2 if kwargs.get("read_twice") else 1) + 10)

    return inference_ids


def _remote(engine_factory, monkeypatch, max_model_len=32768):
    from bielik_one.api import remote_vllm

    reads = []
    monkeypatch.setattr(remote_vllm, "inference_ids", _one_token_per_char(reads))
    monkeypatch.setattr(remote_vllm, "answer_token_ids", lambda tok, row, semantic: [1000, 1001])
    monkeypatch.setattr(server, "answer_token_ids", lambda tok, row, semantic: [1000, 1001])
    fixture = engine_factory(max_model_len)
    return fixture, reads


def test_facts_are_part_of_the_state_that_read_twice_repeats(engine_factory, monkeypatch):
    fixture, reads = _remote(engine_factory, monkeypatch)
    result = fixture.engine.evaluate(request(facts=True))
    extended = "Kosztorys naprawy: 512,40 zł.\n\n" + HEADER["pl"] + "\n- 512,40 zł > 500 zł"
    assert reads == [(extended, False), (extended, True)]  # the length check and the repeated prompt both see them
    assert len(fixture.requests[0]["prompt"]) == 2 * len(extended) + 10
    assert result["answers"]["q"]["prompt"]["variant"] == "read_twice"
    assert result["facts"]["lines"] == ["512,40 zł > 500 zł"]


def test_facts_count_toward_the_read_twice_threshold(engine_factory, monkeypatch):
    fixture, reads = _remote(engine_factory, monkeypatch)
    plain = len("Kosztorys naprawy: 512,40 zł.") + 10
    result = fixture.engine.evaluate(request(facts=True, read_twice_max_input_tokens=plain))
    info = result["answers"]["q"]["prompt"]
    assert (info["variant"], info["reason"]) == ("standard", "input_threshold")
    assert result["facts"]["lines"] == ["512,40 zł > 500 zł"]
    assert HEADER["pl"] in reads[0][0]


def test_without_room_for_the_facts_the_request_is_read_without_them(engine_factory, monkeypatch):
    plain = "Kosztorys naprawy: 512,40 zł."
    fixture, reads = _remote(engine_factory, monkeypatch, max_model_len=len(plain) + 11)
    result = fixture.engine.evaluate(request(facts=True))
    assert result["facts"]["lines"] == []
    assert [state for state, _ in reads][-1] == plain
    assert len(fixture.requests) == 1 and len(fixture.requests[0]["prompt"]) == len(plain) + 10
    assert result["answers"]["q"]["prompt"]["reason"] == "context_limit"


def test_a_state_too_long_without_facts_is_still_refused(engine_factory, monkeypatch):
    fixture, _ = _remote(engine_factory, monkeypatch, max_model_len=10)
    with pytest.raises(server.RequestError, match="context limit"):
        fixture.engine.evaluate(request(facts=True))
    assert fixture.requests == []


def test_negative_amounts_and_us_slash_dates():
    assert lines_of("Saldo: -600 zł, opłata −700 zł, zwrot -€800.", "Czy saldo przekracza 500 zł?") == []
    assert [x.day for x in find_dates("Signed 04/05/2027, delivered 04/25/2027.")] == [date(2027, 4, 5),
                                                                                        date(2027, 4, 25)]
    assert [x.day for x in find_dates("Signed 04/05/2027, delivered 25/04/2027.")] == [date(2027, 5, 4),
                                                                                        date(2027, 4, 25)]


def test_today_needs_the_date_right_after_the_word():
    lines = lines_of("The contract was signed today, and the delivery is due on 5 May 2027.", "Was it on time?")
    assert not any("today" in x for x in lines)
    assert any("today" in x for x in lines_of("Today is Wednesday, 5 May 2027.", "Was it on time?"))


def test_the_german_article_des_is_not_december():
    assert find_dates("Nach Artikel 5 des 2020 erlassenen Gesetzes") == []
    assert [x.day for x in find_dates("Frist: 5. des. 2026")] == [date(2026, 12, 5)]


def test_days_off_share_one_budget_over_sentences_and_years():
    import time

    from bielik_one.prompting.facts import MAX_HOLIDAYS, _holidays

    # 300 distinct years, thousands of short holiday sentences, and many sentences of year-less dates
    rule = "Within 10 working days? Holiday " + " ".join(f"1.1.{y}" for y in range(1900, 2200))
    state = "\n".join(
        [f"Holiday {i % 28 + 1} May {1900 + i % 300}." for i in range(2_000)]
        + ["Holiday " + " ".join(f"{d % 28 + 1} May" for d in range(300)) for _ in range(20)]
        + ["Sent 1 June 2027, due 30 June 2027, 1 July 2027."]
    )
    start = time.perf_counter()
    lines, _ = facts(state, [rule])
    assert time.perf_counter() - start < 1.0
    assert len(lines) <= 40
    assert len(_holidays([state, rule], list(range(1900, 2200)))) <= MAX_HOLIDAYS
