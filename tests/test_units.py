import pytest

from dibs.units import (
    GiB,
    MiB,
    UnitError,
    format_duration,
    format_size,
    parse_duration,
    parse_size,
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
