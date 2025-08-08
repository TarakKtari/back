import io
import pandas as pd
from flask import send_file

def _make_excel(
    headers: list[str],
    example: dict[str, str] | None = None,
    filename: str = "template.xlsx"          
):
    df = (
        pd.DataFrame([example], columns=headers)
        if example else
        pd.DataFrame(columns=headers)
    )
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xls:
        df.to_excel(xls, index=False, sheet_name="Template")
    bio.seek(0)
    return send_file(
        bio,
        as_attachment=True,
        download_name=filename,               
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
