import csv
from pathlib import Path


def split_row_key(row_key: str) -> tuple[str, str, str]:
    parts = row_key.split("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected row key format: {row_key!r}")
    return parts[0], parts[1], parts[2]


def merge_to_long(pa_path: Path, hr_path: Path, out_path: Path) -> None:
    with pa_path.open("r", newline="", encoding="utf-8-sig") as f_pa, hr_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as f_hr, out_path.open("w", newline="", encoding="utf-8") as f_out:
        pa_reader = csv.reader(f_pa)
        hr_reader = csv.reader(f_hr)
        writer = csv.writer(f_out)

        pa_header = next(pa_reader)
        hr_header = next(hr_reader)

        pa_times = pa_header[1:]
        hr_times = hr_header[1:]
        if pa_times != hr_times:
            raise ValueError("Time columns do not match between PA and HR files.")

        writer.writerow(["idno", "visit", "date", "time", "pa", "hr"])

        line_no = 1
        while True:
            pa_row = next(pa_reader, None)
            hr_row = next(hr_reader, None)

            if pa_row is None and hr_row is None:
                break
            if pa_row is None or hr_row is None:
                raise ValueError("Input files have different number of data rows.")

            line_no += 1

            pa_key = pa_row[0]
            hr_key = hr_row[0]
            if pa_key != hr_key:
                raise ValueError(
                    f"Row key mismatch at input line {line_no}: {pa_key!r} != {hr_key!r}"
                )

            idno, visit, date = split_row_key(pa_key)

            pa_values = pa_row[1:]
            hr_values = hr_row[1:]
            if len(pa_values) != len(pa_times) or len(hr_values) != len(hr_times):
                raise ValueError(f"Value/time length mismatch at input line {line_no}.")

            for t, pa_value, hr_value in zip(pa_times, pa_values, hr_values):
                writer.writerow([idno, visit, date, t, pa_value, hr_value])


if __name__ == "__main__":
    data_dir = Path(r"d:\job_files\BLSA\BLSA_data\datarelease\BLSA_Actiheart_Summary_Data")
    # output_dir = Path(r"d:\job_files\BLSA\BLSA_data\datarelease\")
    pa_file = data_dir / "ActivityCountRaw_2016_Jun.csv"
    hr_file = data_dir / "HeartRateRaw_2016_Jun.csv"
    out_file = data_dir / "PA_HR_Raw_2016_Jun_long.csv"

    merge_to_long(pa_file, hr_file, out_file)
    print(f"Done: {out_file}")
