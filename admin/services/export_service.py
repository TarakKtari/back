import base64
import io
from datetime import datetime, timedelta
import os
from fpdf import FPDF
from openpyxl.workbook import Workbook
from flask import send_file, make_response

def export_pdf(images):
    pdf = FPDF()
    for i, img_data in enumerate(images):
        img_data = img_data.split(';base64,')[1]
        img_bytes = base64.b64decode(img_data)
        img_filename = f'temp_image_{i}.png'
        with open(img_filename, 'wb') as img_file:
            img_file.write(img_bytes)

        pdf.add_page()
        pdf.image(img_filename, x=10, y=8, w=190)
        os.remove(img_filename)

    pdf_output = f"analytics_{datetime.now().strftime('%Y-%m-%d')}.pdf"
    pdf.output(pdf_output)
    return send_file(pdf_output, as_attachment=True, download_name=f"analytics_{datetime.now().strftime('%Y-%m-%d')}.pdf")

def download_excel(tables_data):
    output = io.BytesIO()
    wb = Workbook()
    for sheet_name, data in tables_data.items():
        ws = wb.create_sheet(title=sheet_name)
        for row in data:
            ws.append(row)
    wb.remove(wb.active)
    wb.save(output)
    output.seek(0)

    filename = f"analytics_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
    response = make_response(output.getvalue())
    response.headers.set('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response.headers.set('Content-Disposition', 'attachment', filename=filename)

    return response
