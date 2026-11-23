import pytest

from dibs.units import (
    GiB,
    MiB,
    UnitError,
    format_duration,
    format_size,
    format_size_against,
    parse_duration,
    parse_size,
    suggest_mem,
    suggest_size,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("512M", 512 * MiB),
        ("16G", 16 * GiB),
        ("16GB", 16 * GiB),
        ("16GiB", 16 * GiB),
        ("16g", 16 * GiB),
        ("1.5G", 1536 * MiB),
        ("1T", 1 << 40),
        ("64K", 64 << 10),
        ("0", 0),
    ],
)
def test_parse_size(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize("bad", ["", "16", "abc", "-1G", "1.5.5G", "G"])
def test_parse_size_rejects(bad):
    with pytest.raises(UnitError) as err:
        parse_size(bad)
    if bad == "16":
        assert "unit" in str(err.value)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("90s", 90),
        ("20m", 1200),
        ("1h", 3600),
        ("1h30m", 5400),
        ("1.5h", 5400),
        ("2d", 172800),
        ("01:02:03", 3723),
        ("02:30", 150),
        ("5", 300),
    ],
)
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected


@pytest.mark.parametrize("bad", ["", "0", "0s", "-5m", "abc", "1x", "1:2:3:4"])
def test_parse_duration_rejects(bad):
    with pytest.raises(UnitError):
        parse_duration(bad)


@pytest.mark.parametrize(
    "n,expected",
    [
        (6 * GiB, "6G"),
        (int(3.1 * GiB), "3.1G"),
        (1536 * MiB, "1.5G"),
        (512 * MiB, "512M"),
        (0, "0"),
    ],
)
def test_format_size(n, expected):
    assert format_size(n) == expected


@pytest.mark.parametrize(
    "n,other,expected",
    [
        (2044 * MiB, 2 * GiB, "2044M"),  # 2 GiB asked, 4 MiB short: not "2G"
        (8188 * MiB, 8 * GiB, "8188M"),  # an "8G" GPU that has 4 MiB less than 8G
        (3073 * MiB, 3 * GiB, "3073M"),  # used 1 MiB over a 3G reservation: not "3G"
        (3 * GiB + MiB // 2, 3 * GiB, "3073M"),  # over by half a MiB: rounded away, up
        (int(1.08 * GiB), parse_size("1.1G"), "1105M"),  # a typed 1.1G keeps its own figure
        (1 * GiB, 2 * GiB, "1G"),  # already reads differently: as format_size
        (2 * GiB, 2 * GiB, "2G"),  # the same amount: nothing to tell apart
        (0, 512 * MiB, "0"),
    ],
)
def test_format_size_against(n, other, expected):
    assert format_size_against(n, other) == expected


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (45, "45s"),
        (480, "8m"),
        (3600, "1h"),
        (5400, "1h30m"),
        (2 * 86400 + 3 * 3600, "2d3h"),
        (0.4, "0s"),
    ],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


@pytest.mark.parametrize(
    "n,expected",
    [
        (int(3.1 * GiB), 4 * GiB),
        (int(5.2 * GiB), 6 * GiB),
        (100 * MiB, 1 * GiB),
        (0, 1 * GiB),
    ],
)
def test_suggest_size(n, expected):
    assert suggest_size(n) == expected


@pytest.mark.parametrize(
    "peak,expected",
    [
        (300 * MiB, 512 * MiB),  # a quarter over the peak, in steps that print as they are
        (500 * MiB, 768 * MiB),
        (700 * MiB, 1 * GiB),
        (1 * GiB, int(1.5 * GiB)),
        (int(1.4 * GiB), 2 * GiB),
        (int(3.5 * GiB), int(4.5 * GiB)),
        (int(7.1 * GiB), 9 * GiB),
        (0, 256 * MiB),
    ],
)
def test_suggest_mem_is_the_peak_plus_a_quarter(peak, expected):
    # Issue #212: the RAM dibs suggests matches what agents are told to ask (#230): the peak plus about a
    # quarter, not the next whole GiB above it plus 10%.
    from dibs.units import format_size, parse_size

    assert suggest_mem(peak) == expected and suggest_mem(peak) >= peak * 1.25
    assert parse_size(format_size(expected)) == expected


@pytest.mark.parametrize("bad", ["9" * 400, "1e999s", "9" * 400 + "h"])
def test_parse_duration_rejects_infinite(bad):
    with pytest.raises(UnitError):
        parse_duration(bad)


@pytest.mark.parametrize("bad", ["9" * 400 + ":00", "9" * 400 + ":00:00"])
def test_parse_duration_rejects_huge_clock(bad):
    with pytest.raises(UnitError):
        parse_duration(bad)


def test_parse_size_rejects_huge():
    with pytest.raises(UnitError):
        parse_size("9" * 400 + "G")
