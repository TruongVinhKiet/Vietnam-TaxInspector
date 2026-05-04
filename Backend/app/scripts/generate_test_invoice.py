import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_invoice(filename):
    doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    normal_style = styles['Normal']
    
    # Title
    elements.append(Paragraph("HÓA ĐƠN GIÁ TRỊ GIA TĂNG (Bản Thể Hiện)", title_style))
    elements.append(Spacer(1, 12))
    
    # Info
    elements.append(Paragraph("Ký hiệu: 1C23TTA", normal_style))
    elements.append(Paragraph("Số: 0001234", normal_style))
    elements.append(Paragraph("Ngày lập: 02/05/2026", normal_style))
    elements.append(Spacer(1, 24))
    
    # Seller
    elements.append(Paragraph("<b>Đơn vị bán hàng:</b> CÔNG TY TNHH PHẦN MỀM ABC", normal_style))
    elements.append(Paragraph("<b>Mã số thuế:</b> 0101234567", normal_style))
    elements.append(Paragraph("<b>Địa chỉ:</b> Tòa nhà Keangnam, Phạm Hùng, Nam Từ Liêm, Hà Nội", normal_style))
    elements.append(Spacer(1, 12))
    
    # Buyer
    elements.append(Paragraph("<b>Đơn vị mua hàng:</b> CÔNG TY TNHH XYZ", normal_style))
    elements.append(Paragraph("<b>Mã số thuế:</b> 0109876543", normal_style))
    elements.append(Paragraph("<b>Địa chỉ:</b> Quận 1, TP Hồ Chí Minh", normal_style))
    elements.append(Spacer(1, 24))
    
    # Table data
    data = [
        ["STT", "Tên Hàng Hóa, Dịch Vụ", "Đơn Vị Tính", "Số Lượng", "Đơn Giá", "Thành Tiền"],
        ["1", "Phần mềm kế toán", "Gói", "1", "15,000,000", "15,000,000"],
        ["2", "Dịch vụ bảo trì server (Tháng 5)", "Tháng", "1", "2,500,000", "2,500,000"],
        ["3", "Thiết kế website doanh nghiệp", "Website", "1", "20,000,000", "20,000,000"],
        ["4", "Tên miền .vn (1 năm)", "Tên miền", "1", "750,000", "750,000"],
        ["5", "Hosting SSD 50GB (1 năm)", "Gói", "1", "1,200,000", "1,200,000"],
    ]
    
    t = Table(data, colWidths=[30, 200, 70, 50, 80, 90])
    
    # Table Style
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (1,1), (1,-1), 'LEFT'),  # Left align text
        ('ALIGN', (4,1), (-1,-1), 'RIGHT'), # Right align numbers
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    
    elements.append(t)
    elements.append(Spacer(1, 24))
    
    elements.append(Paragraph("<b>Tổng tiền hàng:</b> 39,450,000 VNĐ", normal_style))
    elements.append(Paragraph("<b>Thuế GTGT (10%):</b> 3,945,000 VNĐ", normal_style))
    elements.append(Paragraph("<b>Tổng cộng tiền thanh toán:</b> 43,395,000 VNĐ", normal_style))
    
    doc.build(elements)

if __name__ == "__main__":
    generate_invoice("test_invoice_complex.pdf")
    print("Generated test_invoice_complex.pdf")
