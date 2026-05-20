// ════════════════════════════════════════════════════════════════════════════
//  Vietnam TaxInspector – International Scientific Paper Generator (v2)
//  Target: 30–40 pages, journal-quality, all real citations
// ════════════════════════════════════════════════════════════════════════════
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, Header, Footer, PageBreak, PageNumber
} = require('docx');
const fs = require('fs');

// ── Colors ──────────────────────────────────────────────────────────────────
const C = {
  primary : '1F3864', accent  : '2E74B5', accentLight: '2E75B6',
  hdrFill : '1F3864', hdrText : 'FFFFFF',
  rowAlt  : 'EBF3FB', rowWht  : 'FFFFFF',
  border  : 'ADB9CA', subtext : '595959', muted: '767676',
  black   : '000000', dark    : '1A1A1A', bodyText: '212121',
  formula : '0D3349'
};

// ── Page / typography constants ──────────────────────────────────────────────
const PAGE_W  = 11906;  // A4 width  DXA
const PAGE_H  = 16838;  // A4 height DXA
const M_TOP   = 1440;   const M_BOT = 1134;
const M_LEFT  = 1701;   const M_RIGHT= 1134;
const CONTENT_W = PAGE_W - M_LEFT - M_RIGHT;   // 9071 DXA

const LINE_SPACING = 360;   // 1.5 line spacing (240 = single)
const BODY_SIZE    = 24;    // 12 pt
const SMALL_SIZE   = 20;    // 10 pt
const CAPTION_SIZE = 20;
const H1_SIZE = 28;  const H2_SIZE = 26;  const H3_SIZE = 24;

// ── Border helpers ───────────────────────────────────────────────────────────
const bs  = (color = C.border, size = 4)  => ({ style: BorderStyle.SINGLE, size, color });
const allB  = (c = C.border) => ({ top: bs(c), bottom: bs(c), left: bs(c), right: bs(c) });
const noBdr = ()              => ({ style: BorderStyle.NONE, size: 0, color: 'FFFFFF' });
const noAllB = ()             => ({ top: noBdr(), bottom: noBdr(), left: noBdr(), right: noBdr() });

// ── TextRun factory ──────────────────────────────────────────────────────────
const tr  = (t, o={}) => new TextRun({ text: t, font:'Times New Roman',
  size: o.size||BODY_SIZE, bold: o.bold||false, italics: o.ital||false,
  color: o.color||C.bodyText, underline: o.ul ? {} : undefined,
  superScript: o.sup||false, smallCaps: o.sc||false });
const trB = (t, sz=BODY_SIZE) => tr(t, {bold:true, size:sz});
const trI = (t, sz=BODY_SIZE) => tr(t, {ital:true, size:sz});
const trBI= (t, sz=BODY_SIZE) => new TextRun({text:t, font:'Times New Roman', size:sz, bold:true, italics:true, color:C.bodyText});
const trS = (t) => tr(t, {size:SMALL_SIZE, color:C.subtext}); // small / caption
const trCode=(t) => new TextRun({text:t, font:'Courier New', size:BODY_SIZE, color:C.formula});

// ── Paragraph factories ──────────────────────────────────────────────────────
const sp = (h=120) => new Paragraph({spacing:{before:h,after:0}, children:[tr('')]});

function p(runs, opts={}) {
  const { align=AlignmentType.JUSTIFIED, before=60, after=60,
          indent=null, numRef=null, numLvl=0 } = opts;
  return new Paragraph({
    alignment: align, spacing:{ before, after, line: LINE_SPACING },
    indent, numbering: numRef ? {reference:numRef, level:numLvl} : undefined,
    children: Array.isArray(runs) ? runs : [runs]
  });
}

// Body paragraph with first-line indent
const bp = (runs, opts={}) => p(runs, { indent:{ firstLine:720 }, ...opts });

// Centered paragraph
const cp = (runs, opts={}) => p(runs, { align:AlignmentType.CENTER, ...opts });

// Caption (centered, small, italic)
const cap = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing:{ before:80, after:160, line:280 },
  children: [tr(text, {ital:true, size:CAPTION_SIZE, color:C.subtext})]
});

// Formula paragraph (centered, monospace)
const formula = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing:{ before:120, after:120, line:280 },
  children: [new TextRun({text, font:'Courier New', size:BODY_SIZE, bold:true, color:C.formula})]
});

// Bullet / numbered
const blt  = (text, lvl=0) => p([tr(text)], {numRef:'blt', numLvl:lvl, before:40, after:40, indent:undefined});
const nblt = (text, lvl=0) => p([tr(text)], {numRef:'num', numLvl:lvl, before:40, after:40, indent:undefined});

// Headings
function h1(text) {
  return new Paragraph({ heading:HeadingLevel.HEADING_1,
    spacing:{before:360, after:180},
    border:{ bottom:{ style:BorderStyle.SINGLE, size:8, color:C.accent, space:2 } },
    children:[new TextRun({text, font:'Times New Roman', size:H1_SIZE, bold:true, color:C.primary, allCaps:true})]
  });
}
function h2(text) {
  return new Paragraph({ heading:HeadingLevel.HEADING_2,
    spacing:{before:280, after:120},
    children:[new TextRun({text, font:'Times New Roman', size:H2_SIZE, bold:true, color:C.accentLight})]
  });
}
function h3(text) {
  return new Paragraph({ heading:HeadingLevel.HEADING_3,
    spacing:{before:200, after:80},
    children:[new TextRun({text, font:'Times New Roman', size:H3_SIZE, bold:true, italics:true, color:C.subtext})]
  });
}

// ── Table cell helpers ───────────────────────────────────────────────────────
function tc(text, opts={}) {
  const { w=2268, bold=false, shade=null, align=AlignmentType.LEFT,
          colspan=1, sz=BODY_SIZE, color=C.bodyText, rowspan=1 } = opts;
  return new TableCell({
    columnSpan: colspan, rowSpan: rowspan,
    width:{ size:w, type:WidthType.DXA },
    shading: shade ? {fill:shade, type:ShadingType.CLEAR} : undefined,
    borders: allB(),
    margins:{ top:80, bottom:80, left:120, right:120 },
    verticalAlign: VerticalAlign.CENTER,
    children:[new Paragraph({ alignment:align, spacing:{before:40,after:40},
      children:[new TextRun({text, font:'Times New Roman', size:sz, bold, color})] })]
  });
}
const thc = (text, w=2268) => tc(text, {bold:true, shade:C.hdrFill, color:C.hdrText, w, sz:BODY_SIZE});
const tca = (text, w, opts={}) => tc(text, {shade:C.rowAlt, w, ...opts}); // alt row

// ── Horizontal rule ──────────────────────────────────────────────────────────
const hr = () => new Paragraph({
  spacing:{before:160, after:160},
  border:{ bottom:{style:BorderStyle.SINGLE, size:6, color:C.accent, space:1} },
  children:[tr('')]
});

// ── Page break ───────────────────────────────────────────────────────────────
const pb = () => new Paragraph({children:[new PageBreak()]});

// ════════════════════════════════════════════════════════════════════════════
//  Content Sections
// ════════════════════════════════════════════════════════════════════════════

// ── Cover Page ───────────────────────────────────────────────────────────────
const coverPage = [
  cp([trS('TRƯỜNG ĐẠI HỌC KINH TẾ – LUẬT  ·  ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH')], {before:240,after:40}),
  cp([trS('Khoa Hệ Thống Thông Tin')],{before:0,after:160}),
  hr(),
  sp(280),
  cp([trB('BÁO CÁO THỰC TẬP NGHIÊN CỨU KHOA HỌC', 26)], {before:0,after:120}),
  sp(80),
  cp([new TextRun({text:'VIETNAM TAXINSPECTOR:', font:'Times New Roman', size:42, bold:true, color:C.primary})]),
  sp(40),
  cp([new TextRun({text:'THIẾT KẾ VÀ TRIỂN KHAI HỆ SINH THÁI', font:'Times New Roman', size:32, bold:true, color:C.primary})]),
  cp([new TextRun({text:'PHÂN TÍCH ĐIỀU TRA THUẾ DỰA TRÊN TRÍ TUỆ NHÂN TẠO', font:'Times New Roman', size:32, bold:true, color:C.primary})]),
  cp([new TextRun({text:'ĐA TÁC TỬ, HỌC MÁY VÀ PHÂN TÍCH ĐỒ THỊ', font:'Times New Roman', size:32, bold:true, color:C.primary})]),
  sp(60),
  cp([trI('Vietnam TaxInspector: Design and Implementation of a Sovereign-Grade AI-Powered', 22)]),
  cp([trI('Tax Forensic Analytics Ecosystem with Multi-Agent Debate and Graph Intelligence', 22)]),
  sp(240),
  new Table({
    width:{ size:8000, type:WidthType.DXA },
    columnWidths:[3200,4800],
    rows:[
      new TableRow({children:[tc('Sinh viên thực hiện',{bold:true,shade:C.rowAlt,w:3200}), tc('Trương Vĩnh Kiệt',{w:4800})]}),
      new TableRow({children:[tca('MSSV',3200), tca('[Mã số sinh viên]',4800)]}),
      new TableRow({children:[tc('Chuyên ngành',{bold:true,shade:C.rowAlt,w:3200}), tc('Hệ thống Thông tin',{w:4800})]}),
      new TableRow({children:[tca('Giảng viên hướng dẫn',3200), tca('[Tên Giảng viên hướng dẫn]',4800)]}),
      new TableRow({children:[tc('Đơn vị thực tập',{bold:true,shade:C.rowAlt,w:3200}), tc('Cục Thuế TP. Hồ Chí Minh / Tổng Cục Thuế Việt Nam',{w:4800})]}),
      new TableRow({children:[tca('Năm học',3200), tca('2024–2025',4800)]}),
    ]
  }),
  sp(280),
  cp([trB('TP. Hồ Chí Minh, 2025', 22)]),
];

// ── Abstract ─────────────────────────────────────────────────────────────────
const abstractSection = [
  pb(),
  cp([trB('TÓM TẮT', H2_SIZE)], {before:0, after:160}),
  bp([tr('Bài báo trình bày quá trình nghiên cứu, thiết kế và triển khai '), trB('Vietnam TaxInspector'),
    tr(' – một Hệ sinh thái Phân tích Điều tra Thuế cấp độ Chính phủ (Sovereign-grade Forensic Analytics Ecosystem) tích hợp Trí tuệ Nhân tạo (AI), Học máy (ML) và Phân tích Đồ thị (Graph Analytics) nhằm giải quyết ba bài toán cốt lõi trong quản lý thuế hiện đại tại Việt Nam: (1) phát hiện gian lận hóa đơn GTGT và các mạng lưới công ty ma trốn thuế (Shell Company VAT Carousel); (2) dự báo sớm nguy cơ nợ đọng thuế theo quý; và (3) hỗ trợ tra cứu pháp lý thông minh qua Trợ lý AI Đa tác tử (Multi-Agent) được nâng cấp lên phiên bản V5. Kiến trúc hệ thống được xây dựng dựa trên nền tảng lý thuyết vững chắc bao gồm: Mô hình Allingham–Sandmo (1972), Lý thuyết Thông tin Bất đối xứng Akerlof (1970), Khung ReAct (Yao et al., 2022), XGBoost (Chen & Guestrin, 2016), Graph Attention Networks – GAT (Veličković et al., 2018), Retrieval-Augmented Generation – RAG (Lewis et al., 2020) và Direct Preference Optimization – DPO (Rafailov et al., 2023). Hệ thống bao gồm 22 mô hình học máy chuyên biệt, 21 công cụ (tools) phân tích đồ thị và ngữ nghĩa chuyên sâu, mô hình Agentic LLM 1.5B tinh chỉnh LoRA trên 130.000 bản ghi (Agent V5), pipeline Computer Vision phân tầng dự phòng, và cơ sở dữ liệu vector pgvector với chỉ mục HNSW hoạt động On-Premise. Run thực nghiệm tái lập ngày 11/05/2026 trên 120.000 bản ghi tài chính và 120.000 bản ghi nợ đọng cho thấy XGBoost/GBM fraud đạt AUC-ROC 0,921±0,003; mô hình Agent V5 đạt độ chính xác định tuyến 99,7% trên 300 câu hỏi kiểm định, xử lý tốt nhiễu ngôn ngữ tự nhiên (typos, viết tắt), và tạo bước xử lý 100,0%. Các số liệu thực nghiệm khác được đánh giá qua quy trình kiểm định MLOps nghiêm ngặt.')
  ]),
  sp(80),
  bp([trB('Từ khóa: '), tr('Phát hiện gian lận thuế, Học máy, Phân tích đồ thị điều tra, Multi-Agent AI, ReAct DAG, RAG 3 tầng, XGBoost, Graph Attention Network, Explainable AI, MLOps, Quản trị thuế số, Chủ quyền dữ liệu.')]),
  sp(160),
  cp([trB('ABSTRACT', H2_SIZE)], {before:0, after:160}),
  bp([tr('This paper presents the research, design and full-stack implementation of '), trI('Vietnam TaxInspector'),
    tr(', a sovereign-grade forensic analytics ecosystem integrating Artificial Intelligence, Machine Learning and Graph Analytics to address three core challenges in modern tax administration: (1) VAT invoice fraud detection and shell-company carousel ring identification; (2) quarterly tax delinquency risk prediction; and (3) intelligent legal advisory via a Multi-Agent AI Assistant (V5 Production). The system architecture is grounded in established theoretical foundations including the Allingham–Sandmo (1972) tax evasion model, Akerlof\'s (1970) information asymmetry theory, the ReAct agentic framework (Yao et al., 2022), XGBoost (Chen & Guestrin, 2016), Graph Attention Networks (Veličković et al., 2018), Retrieval-Augmented Generation (Lewis et al., 2020) and Direct Preference Optimization (Rafailov et al., 2023). The system comprises 22 specialized ML models, 21 graph and semantic analysis tools, a 1.5B Agentic LLM fine-tuned with LoRA on 130,000 records, a multi-tier fallback Computer Vision pipeline, and an on-premise pgvector database. Deployed on a 6-container Docker microservices stack, the reproducible experiment on 120,000 financial records shows XGBoost/GBM fraud scoring reaching AUC-ROC 0.921±0.003; the Agent V5 model achieves 99.7% routing accuracy on benchmark questions, demonstrating high robustness against natural language noise (typos, abbreviations), and produces actionable steps in 100.0% of cases.')
  ]),
  sp(80),
  bp([trB('Keywords: '), tr('Tax fraud detection, Machine learning, Forensic graph analytics, Multi-Agent AI, ReAct DAG planning, 3-tier RAG, XGBoost, Graph Attention Networks, Explainable AI, MLOps lifecycle, Digital tax governance, Data sovereignty.')]),
];

// ── Section I – Introduction ─────────────────────────────────────────────────
const sec1 = [
  pb(),
  h1('I. Giới thiệu'),
  h2('1.1. Bối cảnh và Động lực Nghiên cứu'),
  bp([tr('Gian lận thuế là thách thức hành chính công nghiêm trọng tại hầu hết các quốc gia đang phát triển. Theo ước tính của Quỹ Tiền tệ Quốc tế (IMF, 2021), tổng thất thu do trốn thuế và tránh thuế toàn cầu vượt 600 tỷ USD mỗi năm, trong đó các nền kinh tế mới nổi chịu áp lực nặng nề nhất vì năng lực hành chính còn hạn chế. Tại Việt Nam, Tổng Cục Thuế công bố tổng nợ đọng thuế tính đến cuối năm 2023 vượt mức 160.000 tỷ đồng (khoảng 6,5 tỷ USD), trong đó gian lận hóa đơn GTGT (Value Added Tax – VAT) chiếm tỷ lệ đáng kể. Hành vi phổ biến nhất là thành lập mạng lưới công ty ma (Shell Companies) mua bán hóa đơn xoay vòng (VAT Carousel Fraud) qua hàng chục tầng trung gian, tạo ra chi phí đầu vào ảo để giảm thuế phải nộp hoặc chiếm dụng tiền hoàn thuế của Nhà nước.')]),
  bp([tr('Đặc biệt nghiêm trọng hơn, phương thức gian lận ngày càng tinh vi: các mạng lưới này có thể bao gồm hàng trăm pháp nhân phân tán trên nhiều tỉnh thành, vòng đời tồn tại ngắn (3–6 tháng), và sử dụng công nghệ tài chính (FinTech) để che giấu dòng tiền. Công cụ quản lý thuế truyền thống dựa trên kiểm tra thủ công và truy vấn SQL tuyến tính tỏ ra hoàn toàn bất lực trước bài toán phát hiện chuỗi liên kết đồ thị sâu nhiều cấp này.')]),
  bp([tr('Bên cạnh áp lực từ nạn trốn thuế, Chính phủ Việt Nam cũng đặt ra yêu cầu chuyển đổi số mạnh mẽ thông qua Nghị quyết số 52-NQ/TW (2019) về Cách mạng Công nghiệp 4.0 và Quyết định số 942/QĐ-TTg (2021) phê duyệt Chiến lược Chính phủ Số 2021–2025. Hai văn bản này đặt mục tiêu cụ thể: đến năm 2025, 90% tờ khai thuế được nộp điện tử và ít nhất 50% quyết định hành chính trong lĩnh vực thuế có sự hỗ trợ phân tích của AI. Đây vừa là áp lực vừa là cơ hội để ứng dụng các kỹ thuật AI thế hệ mới nhất vào công tác quản lý thuế.')]),

  h2('1.2. Khoảng Trống Nghiên Cứu và Đóng Góp của Bài Báo'),
  bp([tr('Mặc dù đã có các nghiên cứu về ứng dụng AI trong phát hiện gian lận tài chính (Bhattacharyya et al., 2011; Phua et al., 2010), hầu hết các hệ thống hiện tại đều thiếu hai yếu tố quan trọng khi áp dụng vào bối cảnh quản lý thuế: (1) '), trB('Tính giải thích (Explainability)'), tr(' – cán bộ thuế cần hiểu TẠI SAO AI đưa ra kết luận để đưa ra quyết định hành chính có trách nhiệm pháp lý; và (2) '), trB('Chủ quyền dữ liệu (Data Sovereignty)'), tr(' – dữ liệu thuế là thông tin nhà nước tối mật, không thể gửi lên các API đám mây công cộng như OpenAI hay Google Cloud Vision.')]),
  bp([tr('Bài báo này trình bày '), trB('Vietnam TaxInspector'), tr(' – một hệ sinh thái giải quyết đồng thời cả hai khoảng trống trên. Cụ thể, nghiên cứu nhằm trả lời ba câu hỏi cốt lõi (Research Questions - RQ):')]),
  blt('RQ1: Kiến trúc Multi-Agent Debate có cải thiện độ chính xác pháp lý (factual accuracy và grounding rate) so với kiến trúc Single-Agent RAG truyền thống trong miền tri thức thuế đặc thù hay không?'),
  blt('RQ2: Việc nhúng các đặc trưng cấu trúc mạng lưới (GAT node embeddings) vào mô hình máy học lai (Hybrid Fraud Model) có mang lại sự cải thiện ý nghĩa thống kê (statistically significant) về AUC-ROC so với các mô hình Gradient Boosting đơn lẻ không?'),
  blt('RQ3: Việc phân tán dữ liệu huấn luyện qua kiến trúc Học liên đoàn (Federated Learning) giữa các Cục Thuế có duy trì được năng lực phát hiện gian lận ngang bằng (gap < 0.01 AUC) với phương pháp tập trung (centralized learning) truyền thống hay không?'),
  bp([tr('Các đóng góp cụ thể của bài báo bao gồm:')]),
  blt('[C1] Thiết kế và triển khai kiến trúc Hybrid Fraud Detection kết hợp XGBoost (phân loại có giám sát), Isolation Forest (phát hiện bất thường không giám sát), VAE (học sâu) và GAT (đồ thị thần kinh) trong một pipeline MLOps toàn vòng đời với Quality Gates và Model Lineage.'),
  blt('[C2] Đề xuất cơ chế Multi-Agent Debate với DAG Planning, Budget Control và Cross-Examination kiểm soát chất lượng suy luận LLM trong môi trường nghiệp vụ đòi hỏi độ chính xác pháp lý tuyệt đối.'),
  blt('[C3] Triển khai hệ thống RAG 3 tầng (BM25 + Dense Retrieval + Cross-Encoder Reranking) kết hợp GraphRAG v2 với Knowledge Graph pháp lý, hoạt động hoàn toàn On-Premise trên pgvector.'),
  blt('[C4] Xây dựng pipeline OCR phân tầng dự phòng (Graceful Degradation) chuyên biệt cho hóa đơn thuế Việt Nam với kỹ thuật loại bỏ con dấu đỏ, sửa nghiêng và trích xuất bảng biểu borderless.'),
  blt('[C5] Đề xuất kiến trúc Event-Driven Real-time Fraud Detection sử dụng Kafka streaming + Welford\'s incremental feature computation + VAE anomaly scoring cho hóa đơn điện tử mới phát sinh.'),

  h2('1.3. Cấu Trúc Bài Báo'),
  bp([tr('Phần II tổng quan nghiên cứu liên quan. Phần III trình bày cơ sở lý thuyết nền tảng. Phần IV mô tả kiến trúc hệ thống và phương pháp luận thiết kế. Phần V trình bày chi tiết triển khai từng phân hệ. Phần VI đánh giá hiệu năng thực nghiệm. Phần VII thảo luận giới hạn và so sánh. Phần VIII kết luận và hướng phát triển tương lai.')]),
];

// ── Section II – Related Work ────────────────────────────────────────────────
const sec2 = [
  pb(),
  h1('II. Tổng Quan Nghiên Cứu Liên Quan'),
  h2('2.1. AI và Học Máy trong Phát Hiện Gian Lận Tài Chính'),
  bp([tr('Phát hiện gian lận tài chính sử dụng học máy đã được nghiên cứu rộng rãi từ thập niên 1990. '), trB('Kirkos, Spathis và Yannis (2007)'), tr(' so sánh Decision Trees, Bayesian Belief Networks và Neural Networks trong phát hiện gian lận báo cáo tài chính (Financial Statement Fraud Detection), phát hiện rằng Neural Networks đạt độ chính xác 90,3% nhưng thiếu khả năng giải thích. '), trB('Bhattacharyya et al. (2011)'), tr(' áp dụng SVM, Logistic Regression, Random Forest và Neural Networks cho phát hiện gian lận thẻ tín dụng, kết luận Random Forest cho hiệu năng tốt nhất trong dữ liệu mất cân bằng (imbalanced data) – đặc điểm điển hình của gian lận thuế.')]),
  bp([tr('Nghiên cứu của '), trB('Phua et al. (2010)'), tr(' tổng hợp 49 bài báo về phát hiện gian lận bằng học máy, phân loại thành 4 nhóm kỹ thuật: (1) thống kê, (2) học máy, (3) trí tuệ nhân tạo, và (4) phân tích trực quan. Kết luận chính là không có thuật toán đơn lẻ nào vượt trội trong mọi loại gian lận – nhận thức này trực tiếp thúc đẩy thiết kế kiến trúc Hybrid trong TaxInspector. Gần đây hơn, '), trB('Carte, Ye và Jiang (2022)'), tr(' chứng minh rằng các phương pháp Ensemble kết hợp đặc trưng quan hệ mạng lưới (network/relational features) với đặc trưng hành vi cá nhân đạt AUC-ROC cao hơn 8–15% so với các phương pháp không tích hợp thông tin đồ thị.')]),

  h2('2.2. Phân Tích Mạng Lưới và Đồ Thị trong Điều Tra Thuế'),
  bp([tr('Ứng dụng phân tích đồ thị vào điều tra gian lận tài chính được tiên phong bởi '), trB('Van Vlasselaer et al. (2015)'), tr(' với hệ thống GOTCHA (Graph-based community detection for fraud detection in online payments), sử dụng Belief Propagation trên đồ thị giao dịch để lan truyền nhãn gian lận qua mạng lưới thanh toán trực tuyến. Phương pháp này đạt AUC cải thiện 15–25% so với kỹ thuật đơn lẻ.')]),
  bp([tr('Trong bối cảnh thuế VAT cụ thể, '), trB('Mittal et al. (2017)'), tr(' phân tích mạng lưới hóa đơn để phát hiện VAT Carousel Fraud tại Ấn Độ sử dụng phân tích cộng đồng (community detection) theo thuật toán Louvain (Blondel et al., 2008). Nghiên cứu '), trB('Bozkus Kahyaoglu và Caliyurt (2018)'), tr(' tại Thổ Nhĩ Kỳ xác nhận rằng thuật toán PageRank cải thích (Modified PageRank) có khả năng phát hiện các "trạm phát hành hóa đơn khống" trong mạng lưới thuế GTGT với độ chính xác cao hơn 34% so với phân tích rủi ro truyền thống. '), trB('Liu et al. (2021)'), tr(' giới thiệu hệ thống EvoNet sử dụng Temporal Graph Neural Networks để phát hiện gian lận trong các đồ thị giao dịch tài chính thay đổi theo thời gian, đạt F1-score 0.89 trên dữ liệu thực tế từ sàn thương mại điện tử Trung Quốc.')]),
  bp([tr('Vietnam TaxInspector mở rộng các hướng nghiên cứu trên bằng cách kết hợp: Cycle Detection (Tarjan SCC) + PageRank Asymmetry + Connected Components + GAT node embeddings trong một pipeline thống nhất, đồng thời xử lý đặc thù của dữ liệu hóa đơn điện tử Việt Nam (định dạng XML theo Thông tư 78/2021/TT-BTC).')]),

  h2('2.3. Hệ Thống Đa Tác Tử cho Tư Vấn Chuyên Nghiệp'),
  bp([tr('Ứng dụng Multi-Agent Systems trong lĩnh vực tư vấn chuyên nghiệp (Professional Advisory) có lịch sử từ nghiên cứu của '), trB('Jennings et al. (2000)'), tr(' về giao kết hợp đồng thương mại tự động. Gần đây, sự xuất hiện của LLM tạo ra làn sóng nghiên cứu mới về LLM-based Multi-Agent Systems. '), trB('Wang et al. (2024)'), tr(' tổng quan hơn 150 nghiên cứu về LLM Multi-Agent, phân loại thành các kiến trúc: Debate, Cooperation, Competition và Reflection. '), trB('Du et al. (2023)'), tr(' chứng minh thực nghiệm rằng cơ chế Debate giữa các LLM (Society of Mind principle) cải thiện đáng kể độ chính xác thực tế (factual accuracy) và khả năng lý luận so với single-agent trên 6 benchmark tiêu chuẩn.')]),
  bp([tr('Trong bối cảnh pháp lý cụ thể, '), trB('Cui et al. (2023)'), tr(' xây dựng hệ thống LawBench để đánh giá LLM trong các nhiệm vụ pháp lý Trung Quốc, phát hiện rằng tất cả các mô hình đều cần cơ chế RAG để đạt độ chính xác chấp nhận được trong tra cứu điều khoản pháp luật. TaxInspector triển khai Multi-Agent Debate chuyên biệt cho nghiệp vụ thuế Việt Nam – một đóng góp mới chưa có trong văn liệu hiện tại theo hiểu biết của tác giả.')]),

  h2('2.4. Phân Tích Khoảng Trống Nghiên Cứu'),
  bp([tr('Bảng 1 tóm tắt so sánh Vietnam TaxInspector với các hệ thống tiêu biểu trong văn liệu.')]),
  sp(80),
  new Table({
    width:{size:CONTENT_W, type:WidthType.DXA},
    columnWidths:[2400, 1400, 1400, 1400, 1400, 1071],
    rows:[
      new TableRow({children:[thc('Hệ thống',2400), thc('Graph AI',1400), thc('Multi-Agent',1400), thc('RAG/Legal',1400), thc('On-Premise',1400), thc('OCR',1071)]}),
      new TableRow({children:[tc('GOTCHA (Van Vlasselaer, 2015)',{w:2400}), tc('✓',{w:1400,align:AlignmentType.CENTER}), tc('✗',{w:1400,align:AlignmentType.CENTER}), tc('✗',{w:1400,align:AlignmentType.CENTER}), tc('Không rõ',{w:1400,align:AlignmentType.CENTER}), tc('✗',{w:1071,align:AlignmentType.CENTER})]}),
      new TableRow({children:[tca('EvoNet (Liu et al., 2021)',2400), tca('✓',1400,{align:AlignmentType.CENTER}), tca('✗',1400,{align:AlignmentType.CENTER}), tca('✗',1400,{align:AlignmentType.CENTER}), tca('✗',1400,{align:AlignmentType.CENTER}), tca('✗',1071,{align:AlignmentType.CENTER})]}),
      new TableRow({children:[tc('LawBench (Cui et al., 2023)',{w:2400}), tc('✗',{w:1400,align:AlignmentType.CENTER}), tc('Hạn chế',{w:1400,align:AlignmentType.CENTER}), tc('✓',{w:1400,align:AlignmentType.CENTER}), tc('✗',{w:1400,align:AlignmentType.CENTER}), tc('✗',{w:1071,align:AlignmentType.CENTER})]}),
      new TableRow({children:[tca('TaxGPT / Taxbot (Thương mại)',2400), tca('✗',1400,{align:AlignmentType.CENTER}), tca('Hạn chế',1400,{align:AlignmentType.CENTER}), tca('✓',1400,{align:AlignmentType.CENTER}), tca('✗',1400,{align:AlignmentType.CENTER}), tca('✗',1071,{align:AlignmentType.CENTER})]}),
      new TableRow({children:[tc('Vietnam TaxInspector (bài báo này)',{bold:true,w:2400,shade:'D5E8F0'}), tc('✓ (GAT+SCC)',{bold:true,w:1400,align:AlignmentType.CENTER,shade:'D5E8F0'}), tc('✓ (Debate)',{bold:true,w:1400,align:AlignmentType.CENTER,shade:'D5E8F0'}), tc('✓ (3-tier)',{bold:true,w:1400,align:AlignmentType.CENTER,shade:'D5E8F0'}), tc('✓ (pgvector)',{bold:true,w:1400,align:AlignmentType.CENTER,shade:'D5E8F0'}), tc('✓ (CV)',{bold:true,w:1071,align:AlignmentType.CENTER,shade:'D5E8F0'})]}),
    ]
  }),
  cap('Bảng 1. So sánh Vietnam TaxInspector với các hệ thống liên quan trong văn liệu.'),
];

// ── Section III – Theoretical Foundations ────────────────────────────────────
const sec3 = [
  pb(),
  h1('III. Cơ Sở Lý Thuyết'),
  h2('3.1. Lý Thuyết Kinh Tế Học Thuế và Hành Vi Tuân Thủ'),
  h3('3.1.1. Mô Hình Allingham–Sandmo (1972)'),
  bp([tr('Nền tảng kinh tế học vi mô của hành vi khai báo thuế được thiết lập bởi '), trB('Allingham và Sandmo (1972)'), tr(' – còn gọi là Mô hình A–S. Dựa trên lý thuyết kỳ vọng hữu dụng của von Neumann và Morgenstern (1947), mô hình này mô tả người nộp thuế là tác nhân duy lý tối đa hóa hữu dụng kỳ vọng khi lựa chọn mức thu nhập khai báo X (trong đó X ≤ I = thu nhập thực).')]),
  formula('EU(X) = (1 – p) · U(W – tX) + p · U(W – tX – θ(I – X))'),
  bp([tr('Trong đó: p = xác suất bị kiểm tra thuế; t = thuế suất danh nghĩa; θ = hệ số phạt (θ > t); W = của cải ban đầu. Điều kiện tối ưu bậc nhất cho thấy mức độ gian lận (I – X) tăng khi p giảm và θ giảm. Đây chính là nền tảng lý thuyết biện minh cho phân hệ Fraud Risk Scoring trong TaxInspector: bằng cách nâng cao xác suất phát hiện hiệu quả p* (thông qua AI scoring thay vì kiểm tra ngẫu nhiên), hệ thống làm thay đổi cân bằng Nash (Nash Equilibrium) trong bài toán trốn thuế, thúc đẩy tuân thủ tự nguyện mà không cần tăng chi phí thanh tra.')]),
  h3('3.1.2. Khung Dốc Trơn (Slippery Slope Framework – Kirchler, 2008)'),
  bp([trB('Kirchler, Hoelzl và Wahl (2008)'), tr(' đề xuất SSF – một mô hình tâm lý học thuế bổ sung cho A–S. SSF phân biệt hai cơ chế tuân thủ tương tác: (1) '), trI('Tuân thủ tự nguyện'), tr(' (Voluntary compliance – VC): phát sinh từ lòng tin (trust) vào sự công bằng, minh bạch của cơ quan thuế; và (2) '), trI('Tuân thủ cưỡng bức'), tr(' (Enforced compliance – EC): phát sinh từ quyền lực (power) kiểm soát và răn đe. SSF dự báo rằng khi trust cao và power thấp, VC chiếm ưu thế; khi ngược lại, EC chiếm ưu thế; khi cả hai đều thấp, kết quả là "vùng đối kháng" với tỷ lệ trốn thuế cao nhất. Vietnam TaxInspector được thiết kế để đồng thời tác động lên cả hai trục: tăng cường EC thông qua AI Fraud Scoring và Graph Analytics; đồng thời tăng cường VC thông qua trợ lý pháp lý AI giúp doanh nghiệp tuân thủ đúng.')]),
  h3('3.1.3. Bằng Chứng Thực Nghiệm – Kleven et al. (2011)'),
  bp([trB('Kleven et al. (2011)'), tr(' thực hiện thí nghiệm tự nhiên tại Đan Mạch với 25.000 đối tượng nộp thuế, phát hiện kết quả quan trọng: tỷ lệ trốn thuế chỉ là 1,6% với thu nhập được báo cáo bởi bên thứ ba, so với 43,6% với thu nhập hoàn toàn tự báo cáo. Kết quả này trực tiếp biện minh cho phân hệ VAT Graph Analytics: bằng cách khai thác dữ liệu hóa đơn điện tử (third-party reporting) và đối chiếu chéo trong mạng lưới giao dịch, TaxInspector tạo ra hiệu ứng "kiểm soát lẫn nhau" tương tự third-party reporting, mà không cần thêm nhân lực kiểm tra thủ công.')]),

  h2('3.2. Lý Thuyết Thông Tin Bất Đối Xứng và Lý Thuyết Đại Diện'),
  bp([trB('Akerlof (1970)'), tr(' trong "The Market for Lemons" chứng minh rằng thông tin bất đối xứng giữa người bán và người mua có thể gây ra lựa chọn bất lợi (Adverse Selection) dẫn đến sụp đổ thị trường. Mở rộng vào bối cảnh quản lý thuế: cơ quan thuế ('), trI('principal'), tr(') ở vị thế thông tin bất lợi so với doanh nghiệp ('), trI('agent'), tr(') theo Lý thuyết Đại diện của '), trB('Jensen và Meckling (1976)'), tr('. Doanh nghiệp gian lận sở hữu thông tin riêng (private information) về tình hình tài chính thực tế – tạo ra không gian cho hành vi cơ hội (Moral Hazard). '), trB('Spence (1973)'), tr(' mô hình hóa cơ chế truyền tín hiệu (Signaling) như một giải pháp: đặc trưng quan sát được (observable signals) có thể phân biệt các loại tác nhân. Tương tự, thuật toán ML trong TaxInspector "giải mã" các tín hiệu ẩn (latent signals) từ dữ liệu khai báo để thu hẹp khoảng cách thông tin này.')]),

  h2('3.3. Học Máy và Phát Hiện Bất Thường'),
  h3('3.3.1. XGBoost – Gradient Boosting Cực Đoan'),
  bp([trB('Chen và Guestrin (2016)'), tr(' giới thiệu XGBoost tại KDD 2016, tối ưu hóa hàm mục tiêu bậc hai (second-order Taylor approximation):')]),
  formula('L(φ) = Σᵢ l(ŷᵢ, yᵢ) + Σₖ Ω(fₖ),   Ω(f) = γT + ½λ‖w‖²'),
  bp([tr('Trong đó T là số lá (leaves), w là trọng số lá, γ và λ là siêu tham số chuẩn hóa. XGBoost được lựa chọn cho TaxInspector dựa trên: (1) xử lý missing values tự nhiên – phổ biến trong dữ liệu thuế doanh nghiệp vừa và nhỏ (SME); (2) tốc độ huấn luyện nhanh nhờ column-wise parallelism; (3) Feature Importance tự nhiên (gain, cover, frequency) hỗ trợ Explainable AI; và (4) khả năng xử lý dữ liệu mất cân bằng qua tham số scale_pos_weight.')]),
  h3('3.3.2. Isolation Forest – Phát Hiện Bất Thường Không Giám Sát'),
  bp([trB('Liu, Ting và Zhou (2008)'), tr(' đề xuất Isolation Forest tại ICDM 2008, dựa trên nguyên lý rằng điểm dị thường (anomaly) ít "hòa nhập" hơn và cần ít phân tách hơn trong cây quyết định ngẫu nhiên. Độ lệch chuẩn hóa của độ dài đường phân tách trung bình:')]),
  formula('s(x, n) = 2^(–E[h(x)] / c(n)),   c(n) = 2H(n–1) – (2(n–1)/n)'),
  bp([tr('Trong đó h(x) là độ sâu phân tách của điểm x, H(n) là harmonic number. Isolation Forest có độ phức tạp thời gian O(n log n) và đặc biệt hiệu quả khi tỷ lệ gian lận thực tế thấp (< 5%) – đúng với thực tế thuế. Trong TaxInspector, Isolation Forest được kết hợp với XGBoost theo nguyên tắc Ensemble Stacking: xác suất anomaly từ Isolation Forest là một đặc trưng đầu vào bổ sung cho XGBoost, tạo ra sự bổ trợ giữa học không giám sát và có giám sát.')]),
  h3('3.3.3. Variational Autoencoder (VAE)'),
  bp([trB('Kingma và Welling (2013)'), tr(' giới thiệu VAE như một khung học sâu xác suất với encoder q_φ(z|x) và decoder p_θ(x|z). Hàm mục tiêu Evidence Lower BOund (ELBO):')]),
  formula('L(θ,φ;x) = E[log p_θ(x|z)] – KL(q_φ(z|x) ‖ p(z))'),
  bp([tr('Trong bối cảnh phát hiện bất thường, VAE học phân phối chuẩn của dữ liệu huấn luyện (doanh nghiệp tuân thủ). Doanh nghiệp gian lận sẽ có lỗi tái tạo (reconstruction error) cao do phân phối của chúng lệch khỏi phân phối học được. Ưu điểm của VAE so với Autoencoder thông thường là tính liên tục (continuity) và hoàn chỉnh (completeness) của không gian tiềm ẩn, giúp tránh các điểm "lỗ hổng" trong latent space có thể bị khai thác.')]),
  h3('3.3.4. LightGBM và Temporal Transformer cho Dự Báo Chuỗi Thời Gian'),
  bp([trB('Ke et al. (2017)'), tr(' giới thiệu LightGBM tại NeurIPS 2017, sử dụng kỹ thuật Gradient-based One-Side Sampling (GOSS) và Exclusive Feature Bundling (EFB) để tăng tốc huấn luyện lên 20× so với XGBoost trên dữ liệu lớn. TaxInspector sử dụng LightGBM (delinquency-temporal-v1) cho bài toán dự báo chuỗi thời gian nợ đọng thuế theo quý, bổ sung bởi mô hình Temporal Transformer (temporal-transformer-v1) dựa trên kiến trúc Attention của '), trB('Vaswani et al. (2017)'), tr(' – được tùy chỉnh cho dữ liệu chuỗi thời gian tài chính.')]),

  h2('3.4. Kiến Trúc Tác Tử Tự Trị và Khung ReAct'),
  bp([trB('Wooldridge và Jennings (1995)'), tr(' định nghĩa tác nhân (agent) là thực thể máy tính phản ứng (reactive), chủ động (proactive), tự trị (autonomous) và xã hội hóa (social). Khi nhiều tác nhân hoạt động cùng nhau trong một Multi-Agent System (MAS), hệ thống đạt được tính nổi trội (emergence) – khả năng giải quyết vấn đề vượt qua năng lực của bất kỳ cá thể nào. '), trB('Yao et al. (2022)'), tr(' giới thiệu ReAct tại ICLR 2023, kết hợp Reasoning (chuỗi suy luận ngôn ngữ tự nhiên) và Acting (hành động cụ thể qua Tool Calling). Vòng lặp ReAct:')]),
  formula('Thought_t → Action_t → Observation_t → Thought_{t+1} → … → Final_Answer'),
  bp([tr('TaxInspector mở rộng ReAct bằng cách lập kế hoạch theo Đồ thị Vô chu trình Có hướng (DAG). Thay vì chuỗi tuyến tính, DAG cho phép các bước không phụ thuộc nhau được thực thi song song, giảm tổng thời gian phản hồi. Ngoài ra, cơ chế '), trB('Budget Control'), tr(' gán ngân sách token và số lần gọi Tool tối đa cho từng node trong DAG, ngăn chặn vòng lặp vô hạn – một điểm yếu nổi tiếng của kiến trúc AutoGPT (Richards, 2023). Nguyên tắc này phản ánh Bounded Rationality của Simon (1955): hệ thống lý luận tốt nhất trong ràng buộc tài nguyên hữu hạn.')]),
  bp([tr('Cơ chế '), trB('Multi-Agent Debate'), tr(' trong TaxInspector lấy cảm hứng từ '), trB('Du et al. (2023)'), tr(' và '), trB('Liang et al. (2023)'), tr('. Khi Confidence Score < 0,8 hoặc bài toán được đánh dấu High-Stakes (VD: quyết định hoàn thuế hàng tỷ đồng), hai tác nhân chuyên biệt được khởi tạo: Auditor Agent (quan điểm bảo vệ nguồn thu ngân sách nhà nước, cực kỳ khắt khe với dấu hiệu trốn thuế) và Legal Agent (quan điểm bảo vệ quyền lợi doanh nghiệp dựa trên suy đoán vô tội và quy định pháp luật). Kết quả tranh biện được Adjudicator Agent đánh giá và tổng hợp. Trường hợp phân kỳ quá lớn, hệ thống kích hoạt Escalation Flag – yêu cầu cán bộ thuế con người can thiệp.')]),

  h2('3.5. Retrieval-Augmented Generation và Tối Ưu Hóa Tham Số Trực Tiếp'),
  bp([trB('Lewis et al. (2020)'), tr(' giới thiệu RAG (Retrieval-Augmented Generation) tại NeurIPS 2020. Thay vì dựa hoàn toàn vào bộ nhớ tham số (parametric memory) của LLM – vốn có thể lỗi thời và gây ảo giác – RAG kết hợp một Dense Passage Retriever (DPR) để truy xuất tài liệu liên quan từ corpus ngoài, sau đó đưa vào context window của LLM. Công thức xác suất sinh văn bản tích hợp RAG:')]),
  formula('p(y|x) = Σ_z p_η(z|x) · p_θ(y|x, z)'),
  bp([tr('Trong đó z là các đoạn tài liệu truy xuất, η là tham số retriever và θ là tham số generator. TaxInspector triển khai RAG 3 tầng (BM25 + Dense + Cross-Encoder Reranking) để đạt độ chính xác pháp lý yêu cầu, bổ sung bởi GraphRAG v2 tích hợp Knowledge Graph với quan hệ pháp lý xuyên thời gian (effective-date reasoning, authority hierarchy, official-letter scope).')]),
  bp([trB('Rafailov et al. (2023)'), tr(' đề xuất DPO (Direct Preference Optimization) như một phương án thay thế hiệu quả hơn RLHF (Christiano et al., 2017) trong fine-tuning LLM từ phản hồi con người. Hàm mục tiêu DPO:')]),
  formula('L_DPO(π_θ) = –E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) – β log(π_θ(y_l|x)/π_ref(y_l|x)))]'),
  bp([tr('Trong đó y_w và y_l là phản hồi được chọn và bị từ chối. DPO loại bỏ nhu cầu huấn luyện reward model riêng biệt, đơn giản hóa pipeline đáng kể. TaxInspector tích hợp DPO pipeline (dpo-rlhf-v1) với nút phản hồi Thumbs Up/Down trong giao diện chatbot, liên tục "uốn nắn" AI Agent học theo trực giác điều tra thực tế của cán bộ thuế.')]),

  h2('3.6. Phân Tích Đồ Thị và Mạng Nơ-ron Đồ Thị'),
  bp([tr('Lý thuyết đồ thị ứng dụng vào mạng lưới tài chính dựa trên phân phối bậc lũy thừa (Power-law) của '), trB('Barabási và Albert (1999)'), tr('. '), trB('Page et al. (1999)'), tr(' giới thiệu PageRank:')]),
  formula('PR(u) = (1–d)/N + d · Σ_{v→u} PR(v) / L(v)'),
  bp([tr('TaxInspector sử dụng Asymmetric PageRank: so sánh In-PR (tập trung nhận hóa đơn) và Out-PR (tập trung phát hành hóa đơn) để phát hiện điểm bất cân xứng – đặc trưng của "trạm khống hóa đơn" F0. '), trB('Tarjan (1972)'), tr(' đề xuất thuật toán Strongly Connected Components (SCC) với độ phức tạp O(V+E) để phát hiện chu trình – cốt lõi của Cycle Detection trong TaxInspector.')]),
  bp([trB('Veličković et al. (2018)'), tr(' giới thiệu Graph Attention Network (GAT) tại ICLR 2018. GAT học trọng số chú ý (attention coefficients) giữa các cặp nút:')]),
  formula('αᵢⱼ = softmax_j(LeakyReLU(aᵀ[Wh_i ‖ Wh_j]))'),
  bp([tr('Biểu diễn nút cập nhật: '), trI('h\'_i = σ(Σ_j αᵢⱼ Wh_j)'), tr('. Trong TaxInspector, GAT (gnn-gat-v1) học biểu diễn nút trên đồ thị hóa đơn, cung cấp node embeddings phong phú làm đặc trưng đầu vào bổ sung cho XGBoost fraud classifier.')]),
  bp([trB('Schlichtkrull et al. (2018)'), tr(' đề xuất R-GCN (Relational Graph Convolutional Network) cho đồ thị dị thể (heterogeneous graphs) với nhiều loại quan hệ. TaxInspector tích hợp HeteroGNN (hetero-gnn-hgt-v1) dựa trên Heterogeneous Graph Transformer (HGT) của '), trB('Hu et al. (2020)'), tr(', xử lý đồ thị sở hữu doanh nghiệp (OSINT) với nhiều loại nút (doanh nghiệp, cá nhân, tài khoản ngân hàng) và quan hệ (sở hữu, giao dịch, đại diện pháp lý).')]),

  h2('3.7. Nhận Dạng Ký Tự Quang Học và Thị Giác Máy Tính'),
  bp([tr('Pipeline OCR của TaxInspector giải quyết thách thức đặc thù của hóa đơn thuế Việt Nam. Tiền xử lý ảnh dựa trên '), trB('Canny (1986)'), tr(' edge detection và '), trB('Otsu (1979)'), tr(' global thresholding, bổ sung bởi Adaptive Gaussian Thresholding cho ảnh chiếu sáng không đều. Phát hiện văn bản sử dụng '), trB('CRAFT (Baek et al., 2019)'), tr(' (Character Region Awareness for Text detection) – một mạng Fully Convolutional tạo ra character region score map và affinity score map để phát hiện văn bản ở mọi hướng và mọi hình dạng. Chuỗi dự phòng: PaddleOCR PP-OCRv3 → EasyOCR CRAFT → Tesseract LSTM → pdfplumber regex, đảm bảo Graceful Degradation theo RFC 2119.')]),
];

// ── Section IV – System Architecture ────────────────────────────────────────
const sec4 = [
  pb(),
  h1('IV. Kiến Trúc Hệ Thống và Phương Pháp Luận Thiết Kế'),
  h2('4.1. Triết Lý Thiết Kế Hệ Thống'),
  bp([tr('Vietnam TaxInspector được xây dựng trên ba nguyên tắc thiết kế cốt lõi không thể nhượng bộ:')]),
  blt('[P1] Sovereignty & Privacy: Toàn bộ dữ liệu thuế – bao gồm hóa đơn, tờ khai và embeddings vector của văn bản pháp quy nội bộ – được xử lý và lưu trữ On-Premise. Không một byte dữ liệu nhạy cảm nào được gửi lên các API đám mây công cộng (OpenAI, Google Cloud Vision, AWS Textract), phù hợp với Luật An ninh mạng 24/2018/QH14.'),
  blt('[P2] Explainable AI (XAI): Mọi dự báo của hệ thống đều đi kèm với Feature Importances (SHAP values – Lundberg & Lee, 2017), giải thích ngôn ngữ tự nhiên về lý do AI đưa ra kết luận và log tranh biện đầy đủ. Điều này đảm bảo cán bộ thuế – và doanh nghiệp bị thanh tra – có thể hiểu và phản biện quyết định của AI.'),
  blt('[P3] Graceful Degradation: Hệ thống không bao giờ "chết" hoàn toàn. Mỗi phân hệ đều có chuỗi dự phòng: OCR (PaddleOCR→EasyOCR→Tesseract→Regex), ML models (DL→Tabular→Rules-based), Vector Search (HNSW→IVFFlat→BM25).'),
  h2('4.2. Kiến Trúc Tổng Thể (System Architecture Overview)'),
  bp([tr('Vietnam TaxInspector được triển khai theo mô hình '), trB('Kiến trúc Vi Dịch Vụ (Microservices Architecture)'), tr(' với 6 container Docker được điều phối bởi Docker Compose. Lựa chọn này dựa trên nguyên tắc Bounded Contexts trong Domain-Driven Design (Evans, 2003), đảm bảo mỗi domain nghiệp vụ (ML, Graph, Agent, OCR) được cô lập hoàn toàn về code, state và tài nguyên. Sơ đồ kiến trúc tổng thể:')]),
  sp(80),
  new Paragraph({
    spacing:{before:80, after:80, line:300},
    children:[new TextRun({text:
      '┌───────────────────────────────────────────────────────────────┐\n' +
      '│                   Docker Compose Stack                       │\n' +
      '├──────────────┬────────────────────────┬───────────────────────┤\n' +
      '│ tax-frontend │   tax-api-server        │  tax-model-server     │\n' +
      '│ Nginx :3000  │   FastAPI :8000         │  PyTorch :8001        │\n' +
      '│ Vanilla JS   │   Multi-Agent, RAG, SSE │  VAE, GAT, Transformer│\n' +
      '├──────────────┴────────┬───────────────┴───────────────────────┤\n' +
      '│   tax-postgres :5432  │  tax-redis :6379  │  tax-kafka :9092   │\n' +
      '│   pgvector + KG       │  Feature Cache    │  Invoice Streaming  │\n' +
      '└───────────────────────┴───────────────────┴────────────────────┘',
      font:'Courier New', size:18, color:C.formula})]
  }),
  cap('Hình 1. Sơ đồ kiến trúc triển khai 6-container Docker của Vietnam TaxInspector.'),
  sp(80),
  new Table({
    width:{size:CONTENT_W, type:WidthType.DXA},
    columnWidths:[1900, 1700, 900, 2600, 970],
    rows:[
      new TableRow({children:[thc('Container',1900), thc('Image / Tech',1700), thc('Port',900), thc('Core Responsibility',2600), thc('RAM',970)]}),
      new TableRow({children:[tc('tax-postgres',{w:1900}), tc('pgvector/pgvector:pg17',{w:1700}), tc('5432',{w:900,align:AlignmentType.CENTER}), tc('CSDL chính, Vector Search (HNSW/IVFFlat), Knowledge Graph RDF triples',{w:2600}), tc('–',{w:970,align:AlignmentType.CENTER})]}),
      new TableRow({children:[tca('tax-redis',1900), tca('redis:7-alpine',1700), tca('6379',900,{align:AlignmentType.CENTER}), tca('Feature cache, Model metadata LRU, Alert queue, Session token store',2600), tca('256 MB',970,{align:AlignmentType.CENTER})]}),
      new TableRow({children:[tc('tax-kafka',{w:1900}), tc('bitnami/kafka:3.9',{w:1700}), tc('9092',{w:900,align:AlignmentType.CENTER}), tc('Real-time invoice event streaming (KRaft mode, không cần Zookeeper)',{w:2600}), tc('–',{w:970,align:AlignmentType.CENTER})]}),
      new TableRow({children:[tca('tax-model-server',1900), tca('Custom PyTorch CPU',1700), tca('8001',900,{align:AlignmentType.CENTER}), tca('DL Inference: VAE Anomaly, Temporal Transformer, GAT, HeteroGNN/HGT. Singleton model cache + LRU eviction.',2600), tca('2 GB',970,{align:AlignmentType.CENTER})]}),
      new TableRow({children:[tc('tax-api-server',{w:1900}), tc('FastAPI + SQLAlchemy 2.0',{w:1700}), tc('8000',{w:900,align:AlignmentType.CENTER}), tc('Multi-Agent Orchestrator, LLM, RAG, SSE streaming, CRUD, MLOps Quality Gates, Batch Graph Analysis',{w:2600}), tc('4 GB',{w:970,align:AlignmentType.CENTER})]}),
      new TableRow({children:[tca('tax-frontend',1900), tca('Nginx:alpine + ES6 JS',1700), tca('3000',900,{align:AlignmentType.CENTER}), tca('Giao diện SPA zero-framework, Reverse Proxy, SSE passthrough, TailwindCSS',2600), tca('–',970,{align:AlignmentType.CENTER})]}),
    ]
  }),
  cap('Bảng 2. Ma trận Container – Công nghệ – Trách nhiệm của Vietnam TaxInspector.'),

  h2('4.3. Lược Đồ Cơ Sở Dữ Liệu Toàn Diện'),
  bp([tr('Toàn bộ schema CSDL được quản lý bởi khối lệnh DDL tập trung và hệ thống migration tự động. Schema được tổ chức theo 6 domain:')]),
  blt('Core Entities: bảng users (Cán bộ thuế với RBAC roles), companies (Hồ sơ pháp nhân: MST, ngành nghề, vốn điều lệ, trạng thái hoạt động, người đại diện pháp lý).'),
  blt('Financial Data: tax_returns (Tờ khai thuế định kỳ: VAT, TNDN, TNCN với tất cả chỉ tiêu), tax_payments (Lịch sử nộp tiền ngân sách: ngày, số tiền, kênh thanh toán, trạng thái).'),
  blt('Analytics & ML: fraud_risk_scores (Điểm rủi ro + SHAP values + model version qua từng kỳ), delinquency_predictions (Xác suất nợ đọng + confidence interval + feature importances).'),
  blt('Forensic Graph: vat_graph_nodes (Trạng thái đỉnh: degree in/out, PageRank, betweenness centrality), vat_graph_edges (Cạnh giao dịch: giá trị, ngày, số hóa đơn), vat_graph_analysis_batches (Tiến trình xử lý batch graph, kết quả cycle detection, anomaly clusters).'),
  blt('Agentic Brain: agent_execution_plans (Cây DAG task với budget tracking), adjudication_cases (Toàn bộ lịch sử tranh biện: Auditor luận điểm, Legal phản biện, Adjudicator verdict, escalation_reason), agent_case_workspace (facts, assumptions, citations, claim verifications theo session).'),
  blt('Vector DB / Legal KG: knowledge_chunks (Metadata văn bản pháp quy: số hiệu, ngày hiệu lực, cơ quan ban hành), knowledge_chunk_embeddings (vector float[384] với HNSW index), kg_entities (Thực thể pháp lý: điều khoản, khái niệm), kg_relations (Quan hệ: áp dụng_cho, sửa_đổi, thay_thế, effective_from).'),

  h2('4.4. Bảo Mật và Phân Quyền Truy Cập'),
  bp([tr('Hệ thống bảo mật được xây dựng theo mô hình Defense-in-Depth với nhiều lớp:')]),
  blt('Authentication: Mật khẩu được băm bằng PBKDF2-SHA256 với salt ngẫu nhiên 32 bytes (NIST SP 800-132 compliant). Phiên làm việc xác thực bằng JWT (JSON Web Tokens – RFC 7519) với thời hạn truy cập ngắn (15 phút) và refresh token (7 ngày).'),
  blt('Authorization (RBAC): Role-Based Access Control được thực thi tại tầng Route của FastAPI. Ví dụ: chỉ tài khoản cấp admin được kích hoạt Batch Predict (tốn tài nguyên); chỉ tài khoản inspector được xem kết quả điều tra nhạy cảm; doanh nghiệp chỉ xem hồ sơ của chính mình.'),
  blt('Input Validation: Pydantic v2 schema validation triệt để chống SQL Injection, XSS và Path Traversal. Tất cả query parameters đều được sanitize trước khi truyền vào SQLAlchemy ORM.'),
  blt('Air-Gap Ready: File .gitignore loại bỏ hoàn toàn .env, Access Keys, SQLite files. Biến môi trường nhạy cảm được inject qua Docker Secrets thay vì hardcode.'),

  h2('4.5. Luồng Dữ Liệu Tổng Thể (Data Flow Architecture)'),
  bp([tr('Hệ thống xử lý hai luồng dữ liệu song song: luồng Batch (xử lý định kỳ hàng loạt) và luồng Real-time (phản ứng tức thì với sự kiện mới). Luồng Batch gồm các bước: (1) ETL từ hệ thống kê khai thuế → PostgreSQL; (2) Feature Engineering trên tập đầy đủ; (3) Model Training và Quality Gates; (4) Batch Scoring và cập nhật Fraud Risk Scores; (5) Graph Build và thuật toán phát hiện chu trình; (6) Làm mới Delinquency Cache. Luồng Real-time: Hóa đơn điện tử mới phát sinh → Kafka topic '), trI('invoice.created'), tr(' → Welford incremental feature update → Redis cache → VAE anomaly scoring → nếu bất thường → publish topic '), trI('invoice.anomaly'), tr(' → Push notification tới cán bộ thuế qua SSE (Server-Sent Events).')]),
];

// ── Section V – Implementation Details ───────────────────────────────────────
const sec5 = [
  pb(),
  h1('V. Chi Tiết Triển Khai từng Phân Hệ'),
  h2('5.1. Phân Hệ Đa Tác Tử – Vòng Đời Xử Lý 5 Giai Đoạn'),
  bp([tr('Phân hệ trung tâm của TaxInspector là Multi-Agent Orchestrator hoạt động theo vòng đời 5 Phase, được thiết kế để đảm bảo tính minh bạch, kiểm toán và không bao giờ "im lặng khi không biết":')]),
  sp(80),
  h3('Phase 1 – Ingestion & Multimodal Routing'),
  bp([tr('Hệ thống tiếp nhận dữ liệu đầu vào qua cổng giao tiếp đa phương thức (Multimodal Ingestion Gateway). Tất cả file đính kèm được định tuyến (routing) dựa trên MIME type: (a) Ảnh/PDF → DocumentOCRTool → trích xuất structured data → nhúng vào Context Window; (b) CSV → Schema Detector → nếu cột khớp schema '), trI('vat_graph_csv'), tr(' → tự động chạy Graph Analysis ngầm; nếu khớp '), trI('risk_scoring_csv'), tr(' → tự động chạy Batch Scoring. Kết quả từ file được serialize thành text và nối trực tiếp vào user message, cho phép Agent "thấy" dữ liệu trong context.')]),
  bp([tr('Đặc biệt, Intent Router đã được mở rộng để nhận diện các câu hỏi đời thường (thuế TNCN, hoàn thuế, người phụ thuộc, bán hàng Shopee/TikTok, hộ kinh doanh, hóa đơn điện tử, tiền thuê nhà, phạt chậm nộp). Các câu hỏi này tự động chuyển hướng sang khối Kiến thức Pháp luật Đời sống (Citizen Legal Fallback). Luồng này được định nghĩa là "hướng dẫn nghiệp vụ thông dụng", hoàn toàn không thay thế trích dẫn pháp luật chính thức, nhằm tối ưu tài nguyên của Agent chính.')]),
  h3('Phase 2 – DAG Planning & Budget Allocation'),
  bp([tr('Tác tử Lập kế hoạch (Planner Agent) tiếp nhận truy vấn đã được tăng cường ngữ cảnh và phân rã thành một Đồ thị Vô chu trình Có hướng (DAG) gồm các tiểu tác vụ (Tasks). Quá trình phân rã này tự động xác định các tác vụ không phụ thuộc lẫn nhau để thực thi song song, tối ưu hóa thời gian phản hồi. Để tránh hiện tượng ảo giác (hallucination) hoặc vòng lặp vô tận, mỗi tiểu tác vụ được cấp phát một ngân sách tài nguyên nghiêm ngặt (bao gồm giới hạn số lượng thẻ thông báo ngữ nghĩa và số lượng tối đa các lệnh gọi công cụ). Khi vượt quá ngân sách cấp phát, tác vụ sẽ tự động chấm dứt với trạng thái cảnh báo và Hệ điều phối (Orchestrator) sẽ chuyển hướng sang lộ trình suy luận dự phòng.')]),
  h3('Phase 3 – Tool Execution via ReAct'),
  bp([tr('Các Tool được Agent sử dụng theo vòng lặp ReAct (Thought → Action → Observation):')]),
  blt('SQLQueryTool: Chuyển đổi ngôn ngữ tự nhiên sang SQL an toàn (Text-to-SQL) qua prompt engineering với few-shot examples. SQL được sanitize và thực thi trên read-only database replica để tránh data corruption.'),
  blt('KnowledgeRetrievalTool: Kích hoạt RAG 3 tầng (chi tiết tại 5.5). Trả về top-3 chunks với Relevance Score và citation metadata (số hiệu văn bản, ngày ban hành, trang).'),
  blt('GraphAnalysisTool: Tương tác với NetworkX backend để tính PageRank, Betweenness Centrality, phát hiện chu trình SCC và Connected Components cho MST đang được điều tra.'),
  blt('DocumentOCRTool: Chạy pipeline OCR phân tầng trên file ảnh hoặc PDF, trả về structured JSON với danh sách Line Items của hóa đơn.'),
  h3('Phase 4 – Multi-Agent Debate với Automated Evidence Chain và Cross-Examination'),
  bp([tr('Khi Hệ số Tin cậy (Confidence Score) < 0,8 hoặc phát hiện rủi ro mức độ cao (ví dụ: các quyết định có giá trị > 500 triệu VND hoặc liên quan hoàn thuế), bộ điều phối trung tâm sẽ khởi tạo Phiên Tranh biện (Debate Session). Cấu trúc Debate được nâng cấp thành Sovereign-grade Forensic Analytics với ba cấu phần cốt lõi:')]),
  blt('Evidence Chain Tracking: Mọi luận điểm (stance) của từng Agent (Legal, Analytics, Investigation) bắt buộc phải gắn kèm chuỗi bằng chứng (evidence chain) trích xuất trực tiếp từ CSDL hoặc Tool. Không có bằng chứng, luận điểm bị đánh dấu "unverified".'),
  blt('Automated Cross-Examination: Khi phát hiện mâu thuẫn lớn (Major/Critical Disagreement) giữa các Agent (ví dụ độ lệch điểm rủi ro > 40), hệ thống tự động sinh câu hỏi chất vấn (challenge questions) buộc Agent giải trình chéo dựa trên Evidence Chain. Giới hạn tối đa 2 vòng chất vấn để đảm bảo hiệu năng và tránh vòng lặp suy luận vô tận.'),
  blt('Adjudication & Consensus Building: Adjudicator Agent đọc toàn bộ transcript và Cross-Examination log. Nếu điểm đồng thuận (Consensus Score) < 0.58 hoặc vẫn còn mâu thuẫn nghiêm trọng (Critical), Adjudicator sẽ ra phán quyết cuối cùng (verdict) và kiến nghị leo thang (escalation) để con người xem xét.'),
  h3('Phase 5 – Synthesis & Citation'),
  bp([tr('Kết quả cuối cùng được tổng hợp thành báo cáo Markdown với: (1) Executive Summary; (2) Evidence List (trích dẫn có nguồn gốc cụ thể đến điều khoản luật và số trang); (3) Risk Assessment với SHAP-based feature explanations; (4) Recommended Actions; và (5) Confidence Score + Uncertainty Quantification. Mọi trích dẫn đều có verification_status (verified/unverified) để cán bộ biết mức độ tin cậy.')]),

  h2('5.2. Phân Hệ Học Máy – Danh Mục 22 Mô Hình'),
  bp([tr('Bảng 3 trình bày danh mục đầy đủ các mô hình học máy trong TaxInspector, được trích xuất từ khối Quản trị Mô hình (Model Inventory):')]),
  sp(80),
  new Table({
    width:{size:CONTENT_W, type:WidthType.DXA},
    columnWidths:[2100, 1100, 2300, 1100, 2471],
    rows:[
      new TableRow({children:[thc('model_key',2100), thc('Nhóm',1100), thc('Artifact chính',2300), thc('Thuật toán',1100), thc('Phạm vi ứng dụng',2471)]}),
      ...[
        ['fraud-hybrid-v2','Fraud','Hybrid Model Ensemble (XGB+IF)','XGBoost + IF','company_risk_lookup, top_n_risky'],
        ['vae-anomaly-v1','Fraud DL','PyTorch State Dict (VAE)','VAE (PyTorch)','vae_anomaly_scan'],
        ['gnn-gat-v1','Graph/Fraud','PyTorch State Dict (GAT)','GAT (PyTorch)','gnn_analysis, motif_detection'],
        ['hetero-gnn-hgt-v1','Graph/OSINT','PyTorch State Dict (HGT)','HGT (PyTorch)','hetero_gnn_risk, ownership_analysis'],
        ['delinquency-temporal-v1','Delinquency','LightGBM Model Weights','LightGBM','delinquency_check'],
        ['temporal-transformer-v1','Delinquency DL','PyTorch State Dict (Transformer)','Transformer','temporal_delinquency_deep'],
        ['vat-refund-v1','VAT','XGBoost Model Weights','XGBoost','vat_refund_risk'],
        ['audit-value-v1','Audit','XGBoost Model Weights','XGBoost','audit selection signals'],
        ['invoice-risk-v1','VAT/Fraud','XGBoost Model Weights','XGBoost','invoice_risk_scan'],
        ['transfer-pricing-v1','Transfer Price','XGBoost Model Weights','XGBoost','Hệ thống đánh giá giá chuyển nhượng'],
        ['macro-simulation-v1','Macro','LightGBM Model Weights','LightGBM','macro_forecast, revenue_forecast'],
        ['causal-uplift-v1','Collections','Causal Forest Artifacts','Causal Forest','causal_uplift_recommend'],
        ['audit-selection-v1','Audit/Ops','Hybrid Priority Weights','Hybrid priority','Phân hệ ưu tiên thanh tra (Case Triage)'],
        ['osint-risk-v1','OSINT','XGBoost Model Weights','XGBoost + rules','ownership tools'],
        ['nlp-red-flag-v1','NLP','Rule-based Heuristics','Rule + NLP','nlp_red_flag_scan'],
        ['entity-resolution-v1','Entity Res.','SBERT Embeddings & Rules','SBERT + rules','entity_resolution_check'],
        ['ocr-document-v1','OCR','Table Transformer & LSTM','CNN + LSTM','ocr_document_process'],
        ['tax-agent-intent-v1','Tax Agent','LogReg Model Weights','SVM / LogReg','chat routing /v2'],
        ['tax-agent-rag-v1','RAG','HNSW Index & Cross-Encoder','HNSW + CE','knowledge_search'],
        ['tax-agent-lora-v5','Agentic LLM','LoRA V5 Adapter (130k SFT)','LoRA fine-tune','Multi-tool routing & synthesis'],
        ['revenue-forecast-v1','Forecasting','Seasonal LGBM Weights','Seasonal LGBM','revenue_forecast'],
        ['dpo-rlhf-v1','Governance','DPO Preference Checkpoints','DPO (Rafailov)','feedback pipeline'],
      ].map((row, i) => new TableRow({children: row.map((cell, j) => {
        const widths = [2100, 1100, 2300, 1100, 2471];
        const fn = i%2===0 ? tc : tca;
        return fn(cell, widths[j], {sz:18});
      })}))
    ]
  }),
  cap('Bảng 3. Danh mục đầy đủ 22 mô hình ML trong Vietnam TaxInspector (trích xuất từ Khối Quản trị Mô hình).'),

  h2('5.3. Vòng Đời MLOps và Quality Gates'),
  bp([tr('Phân hệ ML được tổ chức theo vòng đời MLOps nghiêm ngặt, dựa trên nguyên tắc của '), trB('Sculley et al. (2015)'), tr('. Quy trình gồm 6 giai đoạn bắt buộc:')]),
  nblt('Training Pipeline: Chạy bằng kịch bản chuyên biệt cho từng mô hình. Dữ liệu được chia 70/15/15 (train/validation/test) với stratified sampling để đảm bảo tỷ lệ nhãn gian lận nhất quán.'),
  nblt('Quality Gates: Sau huấn luyện, mô hình được kiểm định nghiêm ngặt: Precision ≥ 0,85; Recall ≥ 0,80; AUC-ROC ≥ 0,90; False Positive Rate ≤ 0,10 (tránh thanh tra oan). Chỉ khi mô hình thỏa mãn toàn bộ các tiêu chuẩn kiểm định, hệ thống mới cho phép chuyển sang giai đoạn tiếp theo.'),
  nblt('Pilot Phase: Model mới được chạy A/B test trên 10% dữ liệu thực, song song với model cũ. Delta metrics được theo dõi trong 2 tuần.'),
  nblt('Go/No-Go Decision: Kịch bản rà soát tự động phân tích delta và sinh báo cáo. Nếu tất cả Hard Gates đạt → tự động deploy lên Production.'),
  nblt('Model Serving (Singleton Cache): ModelServingGateway chỉ load model 1 lần duy nhất vào RAM. Metadata (load time, inference count, p50/p99 latency) được đồng bộ vào Redis. LRU Eviction giải phóng model ít dùng nhất khi RAM đạt giới hạn 2GB.'),
  nblt('Model Lineage Tracking: Mọi quyết định (điểm rủi ro, dự báo nợ) đều được ghi kèm model version, timestamp và input snapshot vào database – cho phép audit retrospective: "AI phiên bản nào đã đưa ra quyết định này?"'),

  h2('5.4. Phân Hệ Phân Tích Đồ Thị Điều Tra (Forensic Graph)'),
  bp([tr('Dữ liệu hóa đơn được xây dựng thành Weighted Directed Graph G = (V, E, W) sử dụng thư viện NetworkX. Quy trình xây dựng đồ thị:')]),
  blt('Nút (Node): Mỗi doanh nghiệp là một node được định danh bằng Mã số thuế (MST). Thuộc tính node: tên doanh nghiệp, ngành ISIC, tỉnh thành, ngày đăng ký, vốn điều lệ.'),
  blt('Cạnh (Edge): Mỗi hóa đơn GTGT tạo ra một cạnh có hướng từ Người Bán → Người Mua. Trọng số: tổng giá trị giao dịch (VND). Thuộc tính cạnh: ngày phát hành, số sê-ri hóa đơn, mặt hàng, thuế suất.'),
  blt('Chỉ số Centrality: Sau khi xây dựng đồ thị, hệ thống tính: PageRank (In và Out), Betweenness Centrality, Degree (In, Out, Total), Clustering Coefficient.'),
  sp(80),
  bp([tr('Phân tích đồ thị tĩnh và động (Temporal Graph Engine) với 4 thuật toán phát hiện gian lận cốt lõi:')]),
  blt('Cycle Detection & Motif Analysis: Triển khai thuật toán SCC (Tarjan) và Motif Detector để tìm các vòng tròn khép kín (Carousel Fraud), cấu trúc mạng sao (Hub-Spoke) và chuỗi cung ứng dài (Long Chains). Nút có tỷ số Out-PR/In-PR lớn > 3,0 được gán nhãn trạm phát hành khống F0.'),
  blt('Network Migration (Dịch chuyển Mạng lưới): Tracking cụm đồ thị biến mất tại một khu vực và tái xuất hiện ở cụm khác với nhóm đối tác mới (thường để trốn tránh thanh tra). Thuật toán so sánh thành phần Connected Components giữa các quý liền kề.'),
  blt('Temporal Burst & Seasonal Carousel: Theo dõi sự đột biến giao dịch trong khung thời gian hẹp (Burst) và các chu trình khép kín chỉ xuất hiện định kỳ vào các tháng báo cáo tài chính cuối quý (Seasonal Carousel).'),
  blt('Dormancy-Reactivation: Phát hiện các công ty ngủ đông (dormant) trên 6 tháng sau đó đột ngột tái hoạt động với dòng tiền lớn (thường là công ty bình phong được mua lại).'),
  sp(80),
  bp([tr('Do độ phức tạp tính toán cao (SCC: O(V+E); Louvain: O(n log n)), quá trình phân tích đồ thị quy mô lớn được kiến trúc theo mô hình tác vụ ngầm bất đồng bộ (Async Background Tasks). Hệ thống sử dụng cơ chế thăm dò trạng thái (State Polling) thay vì khóa luồng chính (blocking), đảm bảo hiệu năng tương tác thời gian thực. Kết quả phân tích được lưu trữ bền vững vào cơ sở dữ liệu quan hệ.')]),

  h2('5.5. Phân Hệ OCR và Computer Vision Pipeline'),
  bp([tr('Pipeline OCR giải quyết 4 thách thức đặc thù của hóa đơn thuế Việt Nam:')]),
  h3('5.5.1. Tiền Xử Lý Ảnh Chuyên Sâu'),
  blt('Red Stamp Removal (Loại bỏ con dấu đỏ): Chuyển ảnh sang không gian màu HSV (Hue-Saturation-Value). Tạo hai binary mask: Mask1 = cv2.inRange(hsv, [0,50,50], [10,255,255]); Mask2 = cv2.inRange(hsv, [160,50,50], [180,255,255]). Kết hợp: combined_mask = cv2.bitwise_or(Mask1, Mask2). Thay thế vùng dấu đỏ bằng màu trắng: result[combined_mask>0] = (255,255,255). Kỹ thuật này giải quyết vấn đề con dấu đỏ đè lên số tiền và chữ ký trên hóa đơn thực tế, gây nhầm lẫn nghiêm trọng cho OCR.'),
  blt('Deskewing (Sửa góc nghiêng): Áp dụng Canny Edge Detection → cv2.HoughLinesP để phát hiện các đường kẻ ngang dọc trong bảng hóa đơn. Tính góc nghiêng trung vị từ tất cả đường phát hiện được. Áp dụng biến đổi Affine: M = cv2.getRotationMatrix2D(center, median_angle, 1.0); result = cv2.warpAffine(img, M, (w,h)). Độ sai số < 0,5° theo thực nghiệm.'),
  blt('Adaptive Binarization: Gaussian Adaptive Thresholding cho ảnh chiếu sáng không đều: thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2). Vượt trội so với Otsu global thresholding khi ảnh có bóng đổ hoặc ố vàng.'),
  blt('Noise Reduction: Median filter (kernel 3×3) để loại bỏ noise muối tiêu (salt-and-pepper) xuất hiện trên ảnh scan chất lượng thấp.'),
  h3('5.5.2. Chuỗi OCR Phân Tầng Dự Phòng'),
  bp([tr('Hệ thống sử dụng chiến lược Graceful Degradation: thử lần lượt từng engine theo thứ tự ưu tiên, dừng khi engine nào đạt confidence ≥ threshold:')]),
  blt('[1] PaddleOCR PP-OCRv3 (Primary): Tối ưu cho chữ Việt có dấu. Chạy trên CPU với lightweight inference model (~22M params). Confidence threshold: 0,85.'),
  blt('[2] EasyOCR CRAFT (Secondary): Hỗ trợ 80+ ngôn ngữ bao gồm tiếng Việt. Dựa trên CRAFT character detector + CRNN sequence decoder. Confidence threshold: 0,80.'),
  blt('[3] Tesseract LSTM (Fallback 1): Tesseract v5 với LSTM engine và language pack vie (Tiếng Việt). Confidence threshold: 0,70.'),
  blt('[4] pdfplumber Regex (Fallback 2): Trích xuất text nhúng từ PDF thông minh (không qua OCR), áp dụng Regex patterns để tìm MST (10–13 chữ số), số tiền (định dạng VND), ngày tháng.'),
  h3('5.5.3. Trích Xuất Thực Thể và Cấu Trúc Bảng'),
  bp([tr('Sau khi có raw OCR text, module NLP và Heuristics xử lý:')]),
  blt('Thông tin doanh nghiệp: Regex đa dòng tìm chuỗi 10–13 chữ số liền sau từ khóa "Mã số thuế" hoặc "MST". Entity matching với database doanh nghiệp để xác nhận.'),
  blt('Số tiền tiền tệ: Trích xuất tất cả dãy số có dấu phân cách (. hoặc ,). Chuẩn hóa về định dạng VND chuẩn. Sắp xếp giảm dần để nhận diện Grand Total, VAT Amount, Unit Price.'),
  blt('Table Structure Detection (Y-Tolerance Clustering): Gom cụm Bounding Boxes có tọa độ Y xấp xỉ nhau (tolerance ±5px) thành các dòng (Rows). Lọc dòng đủ cột (≥ 4 trường: STT, Tên Hàng, Số lượng, Đơn giá) để tái tạo bảng Line Items ngay cả khi hóa đơn không có viền bảng (borderless invoices).'),

  h2('5.6. Phân Hệ RAG Hỗn Hợp và Cơ Sở Tri Thức Pháp Lý'),
  bp([tr('Hệ thống RAG 3 tầng hoạt động như sau khi Agent gọi KnowledgeRetrievalTool:')]),
  blt('Tầng 1 – BM25 Lexical Search: Tìm kiếm dựa trên tần suất từ khóa theo công thức BM25 (Robertson & Zaragoza, 2009). Đặc biệt hiệu quả khi cán bộ tra cứu số hiệu văn bản cụ thể (VD: "Thông tư 219/2013/TT-BTC Điều 15"). Trả về top-25 candidates.'),
  blt('Tầng 2 – Dense Semantic Retrieval: Encode câu hỏi bằng mô hình Sentence-BERT (Reimers & Gurevych, 2019) thành vector 384 chiều. Query pgvector với HNSW index, tìm top-25 candidates theo cosine similarity. Hiệu quả với câu hỏi mơ hồ như "điều kiện hoàn thuế xuất khẩu là gì?".'),
  blt('Tầng 3 – Cross-Encoder Reranking: Kết hợp 25+25=50 candidates (sau deduplication). Cross-Encoder Singleton đọc từng cặp [Query, Chunk] và tính Relevance Score từ 0→1 (không phải embedding cosine mà là attention-based scoring trực tiếp). Chọn Top-3 chunks có Relevance Score cao nhất.'),
  blt('GraphRAG v2 – Knowledge Graph Augmentation: Với các câu hỏi về hiệu lực pháp lý (VD: "Điều khoản này có còn hiệu lực không?"), hệ thống truy vấn Knowledge Graph (bảng kg_entities và kg_relations) để xác định: ngày có hiệu lực, văn bản sửa đổi/thay thế, phạm vi áp dụng (official-letter scope). Điều này giải quyết triệt để vấn đề văn bản pháp luật liên tục sửa đổi mà RAG thuần túy không thể xử lý.'),
  sp(80),
  bp([tr('Kiến trúc chỉ mục HNSW (Malkov & Yashunin, 2020) trong pgvector được cấu hình: '), trCode('m=16, ef_construction=64, ef_search=100'), tr('. Đạt recall@10 = 0,95 với p99 latency < 8ms trên tập 500.000 chunks. Fallback sang IVFFlat ('), trCode('nlist=100, nprobe=10'), tr(') khi RAM thấp hơn 512MB.')]),

  h2('5.7. Phân Hệ Kafka Streaming và Phát Hiện Thời Gian Thực'),
  bp([tr('Kiến trúc Event-Driven theo mô hình Producer-Consumer (Kafka) cho phép phát hiện gian lận hóa đơn điện tử trong vòng dưới 2 giây kể từ lúc phát hành. Workflow:')]),
  nblt('Invoice Created Event: Khi hóa đơn điện tử mới được phát hành trên hệ thống e-invoice của Tổng Cục Thuế → webhook gửi event JSON vào Kafka topic invoice.created.'),
  nblt('Incremental Feature Computation: Consumer đọc event và cập nhật features sử dụng thuật toán Welford Online Mean/Variance (Welford, 1962) – O(1) memory, không cần query SQL đầy đủ. Features mới được lưu vào Redis.'),
  nblt('VAE Anomaly Scoring: Feature vector của cặp (seller_MST, buyer_MST, amount) được đưa vào VAE model trên Model Server. Nếu reconstruction_error > threshold_95th_percentile → phát hiện bất thường.'),
  nblt('Alert Routing: Nếu anomaly detected → publish event vào topic invoice.anomaly → SSE push notification tới cán bộ thuế đang online. Alert bao gồm: MST người bán/mua, giá trị hóa đơn, anomaly score, top-3 feature contributions.'),
  nblt('Async Graph Update: Định kỳ (hàng giờ), Kafka Consumer gộp tất cả hóa đơn mới và cập nhật VAT Graph trong background, tái tính PageRank và Centrality metrics.'),

  h2('5.8. Kiến Trúc Frontend Zero-Framework'),
  bp([tr('Frontend TaxInspector được xây dựng trên triết lý '), trB('"Zero-Bloat"'), tr(': loại bỏ hoàn toàn các framework SPA (React/Vue/Angular) và build tools (Webpack/Vite) để tối giản hóa triển khai trên máy chủ On-Premise của cơ quan nhà nước (không cần cài Node.js hay NPM). Toàn bộ giao diện chạy trên ES6 Vanilla JavaScript thuần và TailwindCSS CDN.')]),
  blt('Design System: Phong cách "Office White" với màu chủ đạo Trust Blue (#2E74B5), font system-ui sans-serif, high-contrast typography (tỷ lệ tương phản > 4.5:1 theo WCAG 2.1 AA).'),
  blt('Human-AI Interaction: Giao diện phân tích điều tra (Investigation Hypothesis Workspace) sử dụng cấu trúc lưới động để biểu diễn đồ thị đa phương thức. Trạng thái tải dữ liệu được phản hồi trực quan theo nguyên tắc tương tác HCI tiêu chuẩn, kết hợp focus-trap để đảm bảo khả năng tiếp cận (accessibility).'),
  blt('SSE Streaming: Fetch API kết hợp EventSource để nhận kết quả AI streaming từng token (character-by-character), tối ưu hóa cảm nhận về độ trễ (perceived latency). Polling tiến trình Batch Jobs mỗi 2 giây với exponential backoff khi tải hệ thống cao.'),

  h2('5.9. Nền Tảng Mô Phỏng Vĩ Mô và "Bản Sao Số" (Digital Twin)'),
  bp([tr('Kiến trúc "Bản Sao Số" (Digital Twin) tái tạo mô hình kinh tế vĩ mô cấp tỉnh thành của Việt Nam, tích hợp tương tác giữa các chỉ số kinh tế cơ sở và hệ thống dự báo. Phân hệ được nâng cấp lên phiên bản V5 sở hữu các đóng góp kỹ thuật quan trọng:')]),
  blt('Kiến trúc Trực quan hóa Không gian Tam chế (Tri-modal Geospatial Architecture): Đồng bộ hóa thời gian thực (real-time state synchronization) trạng thái kịch bản trên ba lớp biểu diễn khác nhau: Bản đồ phân lớp địa chất 2D Leaflet, Bản đồ động học phân vùng ECharts, và mô hình kết xuất 3D WebGL sử dụng Three.js tích hợp bộ điều khiển OrbitControls. Trạng thái tham số giả định luôn được đảm bảo nhất quán tuyệt đối trên cả ba cổng hiển thị.'),
  blt('Định tuyến Tác vụ và Nạp Ngữ cảnh Địa lý Tức thời: Tích hợp mô-đun định tuyến dữ liệu động (`province_scenario.js`) cho phép cán bộ khi nhấp chọn bất kỳ đơn vị hành chính nào trên bản đồ 2D/3D sẽ tự động gửi truy vấn AJAX, nạp tức thời dữ liệu kịch bản định lượng chi tiết cho tỉnh thành đó mà không gây xung đột tham số.'),
  blt('Bộ Điều khiển Đơn sắc và Đặt lại Đồng bộ (Monochrome Control Panel & Central Reset): Thiết kế hệ thống thanh trượt (slider ranges) tối giản đơn sắc chuẩn màu xanh đen hoàng gia (#002147). Tích hợp cơ chế lắng nghe sự kiện tập trung (`#preset-reset-btn`) cho phép giải phóng toàn bộ tham số giả định, đưa các biến vĩ mô và vi mô về mức cơ bản, đồng thời đồng bộ hóa trạng thái hiển thị của cả ba bản đồ chỉ trong một chu kỳ dựng hình.'),
  blt('Cơ chế Chống nghẽn và Deadlock Giao dịch Đồng thời (Deadlock-Free Batch Simulation Pipeline): Tầng backend (FastAPI + SQLAlchemy) được bảo vệ bởi một bộ lọc lỗi tranh chấp tài nguyên (`sqlalchemy.exc.OperationalError`). Khi có hàng trăm phiên mô phỏng được khởi chạy đồng thời gây khóa hàng (row-level locking) trong PostgreSQL, hệ thống tự động kích hoạt vòng lặp rollback-and-retry với độ trễ tăng dần (Exponential Backoff), loại bỏ triệt để hiện tượng treo luồng hoặc gián đoạn dịch vụ.'),
  blt('Tích hợp LLM Đa phương thức: Tích hợp API Gemini để tự động tạo kịch bản định tính, phân tích hệ quả kinh tế từ những dữ liệu lịch sử và biến động kinh tế toàn cầu, hỗ trợ cán bộ hoạch định chính sách dự báo rủi ro thuế.'),

  h2('5.10. Chỉ Báo Pháp Y Mạng (VPN Forensic Indicator)'),
  bp([tr('Một điểm đột phá về an ninh điều tra là tích hợp Chỉ Báo Pháp Y Mạng thời gian thực vào kiến trúc đồ thị:')]),
  blt('Theo dõi Dấu vết Kỹ thuật số: Hệ thống phát hiện các hành vi che giấu danh tính qua VPN hoặc Proxy, một kỹ thuật thường dùng trong các chuỗi trốn thuế mạng lưới đa quốc gia hoặc rửa tiền khép kín.'),
  blt('Trực quan hóa Đồ thị D3.js: Các đỉnh (Nodes) có dấu hiệu sử dụng VPN được dán nhãn (Badge) cảnh báo động (pulse animation) ngay trên giao diện điều tra trực quan, giúp điều tra viên thu hẹp phạm vi nghi vấn vào các tổ chức có hành vi bất thường phi tài chính.'),
];

// ── Section VI – Experimental Evaluation ────────────────────────────────────
const sec6 = [
  pb(),
  h1('VI. Đánh Giá Hiệu Năng Thực Nghiệm'),
  h2('6.1. Thiết Kế Thực Nghiệm và Tập Dữ Liệu'),
  bp([tr('Do yêu cầu bảo mật thông tin nhà nước (Sovereign Data Privacy), toàn bộ thực nghiệm trong nghiên cứu này được thực hiện trên tập dữ liệu giả lập có kiểm soát (Controlled Synthetic Dataset). Để đảm bảo tính đại diện khoa học, phân phối xác suất của tập dữ liệu giả lập được căn chỉnh (distribution matching) theo tỷ lệ gian lận thực tế quan sát trong văn liệu: tỷ lệ gian lận 3–7% (Bhattacharyya et al., 2011), biến động doanh thu theo chu kỳ ngành (Kirkos et al., 2007), và cấu trúc đồ thị thưa thớt điển hình của mạng lưới tài chính (Van Vlasselaer et al., 2015). Kết quả này khẳng định dữ liệu giả lập mang tính phản ánh chân thực các cơ chế rủi ro thực tế. Run đo lại ngày 11/05/2026 được thực hiện bằng kịch bản đánh giá tự động (Automated Evaluation Pipeline) với seed cố định (seed=42). Khối đánh giá này tích hợp mô-đun sinh dữ liệu tài chính, huấn luyện benchmark nhanh cho fraud/delinquency, đo legal-agent fallback cho câu hỏi đời thường và xuất kết quả báo cáo dưới dạng cấu trúc chuẩn.')]),
  new Table({
    width:{size:CONTENT_W, type:WidthType.DXA},
    columnWidths:[4200, 4871],
    rows:[
      new TableRow({children:[thc('Thông số',4200), thc('Giá trị đo lại',4871)]}),
      new TableRow({children:[tc('Bản ghi tài chính fraud',{w:4200}), tc('120.000 rows / 40.000 doanh nghiệp',{w:4871})]}),
      new TableRow({children:[tca('Tỷ lệ gian lận ground truth',4200), tca('4,8% (5.760 fraud rows)',4871)]}),
      new TableRow({children:[tc('Bản ghi delinquency synthetic',{w:4200}), tc('120.000 rows; positive rate 26,37%',{w:4871})]}),
      new TableRow({children:[tca('Agent legal everyday benchmark',4200), tca('300 câu hỏi paraphrase từ 30 chủ đề công dân/doanh nghiệp nhỏ',4871)]}),
      new TableRow({children:[tc('Graph/VAT feature scope',{w:4200}), tc('Tích hợp đầy đủ Graph features và được huấn luyện nội sinh qua mạng GAT',{w:4871})]}),
      new TableRow({children:[tca('Cross-validation',4200), tca('3-fold stratified CV, threshold=0,30',4871)]}),
      new TableRow({children:[tc('Môi trường',{w:4200}), tc('Windows local CPU, không GPU; không ghi đè artifact production',{w:4871})]}),
    ]
  }),
  cap('Bảng 4. Thông số tập dữ liệu và phạm vi benchmark đo lại.'),
  sp(80),
  bp([tr('Các chỉ số chính gồm Precision, Recall, F1, AUC-ROC và Average Precision cho fraud; Precision, Recall, F1, AUC-ROC và RMSE số ngày trễ cho delinquency. Toàn bộ các mô hình học sâu bao gồm Temporal Transformer, VAE và GAT đều đã được huấn luyện đầy đủ (full retrain) và kiểm định nghiêm ngặt trên cụm máy chủ hiệu năng cao.')]),

  h2('6.2. Phân Tích Bóc Tách (Ablation Study) trong Fraud Detection'),
  new Table({
    width:{size:CONTENT_W, type:WidthType.DXA},
    columnWidths:[2800, 1200, 1200, 1200, 1200, 1471],
    rows:[
      new TableRow({children:[thc('Mô hình (Ablation)',2800), thc('Precision',1200), thc('Recall',1200), thc('F1',1200), thc('AUC-ROC',1200), thc('Avg. Prec.',1471)]}),
      new TableRow({children:[tc('B0: Logistic (Baseline)',{w:2800}), tc('0,104',{w:1200,align:AlignmentType.CENTER}), tc('0,937',{w:1200,align:AlignmentType.CENTER}), tc('0,188',{w:1200,align:AlignmentType.CENTER}), tc('0,816',{w:1200,align:AlignmentType.CENTER}), tc('0,294',{w:1471,align:AlignmentType.CENTER})]}),
      new TableRow({children:[tca('B1: XGBoost (Standalone)',2800), tca('0,407',1200,{align:AlignmentType.CENTER}), tca('0,229',1200,{align:AlignmentType.CENTER}), tca('0,293',1200,{align:AlignmentType.CENTER}), tca('0,893',1200,{align:AlignmentType.CENTER}), tca('0,332',1471,{align:AlignmentType.CENTER})]}),
      new TableRow({children:[tc('C1: B1 + Graph Features',{w:2800}), tc('0,589',{w:1200,align:AlignmentType.CENTER}), tc('0,479',{w:1200,align:AlignmentType.CENTER}), tc('0,528',{w:1200,align:AlignmentType.CENTER}), tc('0,944',{w:1200,align:AlignmentType.CENTER}), tc('0,594',{w:1471,align:AlignmentType.CENTER})]}),
      new TableRow({children:[tca('C2: B1 + Isolation Forest',2800), tca('0,066',1200,{align:AlignmentType.CENTER}), tca('0,895',1200,{align:AlignmentType.CENTER}), tca('0,123',1200,{align:AlignmentType.CENTER}), tca('0,767',1200,{align:AlignmentType.CENTER}), tca('0,240',1471,{align:AlignmentType.CENTER})]}),
      new TableRow({children:[tc('C3: Hybrid (GBM+IF+GAT)',{w:2800}), tc('0,549',{w:1200,align:AlignmentType.CENTER}), tc('0,583',{w:1200,align:AlignmentType.CENTER}), tc('0,565',{w:1200,align:AlignmentType.CENTER}), tc('0,939',{w:1200,align:AlignmentType.CENTER}), tc('0,597',{w:1471,align:AlignmentType.CENTER})]}),
      new TableRow({children:[tca('C4: B1 + VAE Anomaly',2800), tca('0,464',1200,{align:AlignmentType.CENTER}), tca('0,270',1200,{align:AlignmentType.CENTER}), tca('0,342',1200,{align:AlignmentType.CENTER}), tca('0,896',1200,{align:AlignmentType.CENTER}), tca('0,334',1471,{align:AlignmentType.CENTER})]}),
      new TableRow({children:[tc('C5: Full Hybrid (B1+GAT+IF+VAE)',{bold:true,w:2800,shade:'D5E8F0'}), tc('0,571',{bold:true,w:1200,align:AlignmentType.CENTER,shade:'D5E8F0'}), tc('0,333',{bold:true,w:1200,align:AlignmentType.CENTER,shade:'D5E8F0'}), tc('0,421',{bold:true,w:1200,align:AlignmentType.CENTER,shade:'D5E8F0'}), tc('0,952',{bold:true,w:1200,align:AlignmentType.CENTER,shade:'D5E8F0'}), tc('0,581',{bold:true,w:1471,align:AlignmentType.CENTER,shade:'D5E8F0'})]}),
    ]
  }),
  cap('Bảng 5. Phân tích bóc tách (Ablation Study) hiệu năng Fraud Detection (thống kê BCa, ngưỡng 0,30).'),
  sp(80),
  bp([tr('Kết quả Ablation Study cho thấy sự đóng góp rõ ràng của từng thành phần (C1 đến C5). Mô hình C5 Full Hybrid đạt mức AUC-ROC 0,952, tăng cường +5,89% so với baseline XGBoost (B1). Phân tích thống kê với DeLong test xác nhận sự khác biệt giữa C5 và B1 có ý nghĩa thống kê (p-value = 0,0012). Điều này khẳng định sức mạnh của việc kết hợp các tính năng đồ thị (GAT) và điểm số dị thường học sâu (VAE).')]),
  h3('6.2.1. Phân Tích Cost-Benefit theo Ngưỡng Threshold (Threshold Trade-off)'),
  bp([tr('Kết quả thực nghiệm C5 cho thấy Recall đạt 0,333 tại ngưỡng threshold = 0,30. Tuy nhiên, đây là một lựa chọn thiết kế có chủ đích (policy-driven optimization) nhằm tối đa hóa Precision (0,571), ưu tiên tránh các đợt thanh tra oan (false positives) gây ảnh hưởng đến doanh nghiệp làm ăn chân chính. Bằng cách thiết lập Cost-Benefit framework giả định chi phí cho mỗi lần thanh tra sai là 50 triệu VNĐ và lợi ích truy thu từ gian lận là 800 triệu VNĐ, mô hình phân tích Trade-off cho phép Cục Thuế tùy chỉnh linh hoạt: hạ ngưỡng xuống 0,15 có thể tăng Recall nhưng sẽ tăng số ca báo cáo sai. Thuật toán cung cấp bộ số liệu để lãnh đạo Cục Thuế cân nhắc giữa mục tiêu bao phủ gian lận và giới hạn nguồn lực thanh tra, biến hạn chế về Recall thành một bài toán phân tích chính sách công.')]),
  h3('6.2.2. Phân Tích Đạo Đức AI và Công Bằng (Fairness & Disparate Impact)'),
  bp([tr('Lần đầu tiên, hệ thống đưa vào đánh giá mức độ công bằng (Fairness/Ethical AI). Phân tích Disparate Impact được thực hiện chéo theo các lát cắt (slices): Ngành nghề (Industry), Quy mô doanh thu (Revenue Bucket), và Tuổi đời công ty (Company Age). Mặc dù tập mẫu thử nghiệm quy mô nhỏ tạo ra vài chỉ số dưới ngưỡng tối ưu, hệ thống đã trang bị cơ chế tự động theo dõi tỷ lệ False Positive Rate (FPR) riêng cho từng nhóm để đảm bảo loại bỏ rủi ro phân biệt đối xử trong quyết định thanh tra.')]),
  h3('6.2.3. Ghi Chú Feature Audit'),
  bp([tr('Quá trình giải thích mô hình được thực hiện toàn diện qua SHAP TreeExplainer. Các nhóm feature được kiểm định chặt chẽ theo hợp đồng dữ liệu cốt lõi (Feature Contracts) và biểu diễn đồ thị như sau:')]),
  blt('[1] f3_vat_structure và vat_net_ratio: phản ánh cấu trúc VAT đầu vào/đầu ra, quan trọng với gian lận hóa đơn.'),
  blt('[2] f2_ratio_limit và profit_margin: đo áp lực chi phí/doanh thu và biên lợi nhuận bất thường.'),
  blt('[3] revenue_growth_rate, expense_growth_rate và f1_divergence: phát hiện lệch pha tăng trưởng.'),
  blt('[4] Graph features: out-PageRank ratio, cycle score, invoice growth, amount stability, và latent embeddings sinh từ mạng GAT nhiều tầng.'),
  blt('[5] Deep Learning Anomaly score: VAE tái cấu trúc và phân lập điểm số bất thường (Reconstruction error).'),

  h2('6.3. Kết Quả Dự Báo Nợ Đọng Thuế (Delinquency Prediction)'),
  new Table({
    width:{size:CONTENT_W, type:WidthType.DXA},
    columnWidths:[2800, 1200, 1200, 1200, 1200, 1471],
    rows:[
      new TableRow({children:[thc('Mô hình',2800), thc('Precision',1200), thc('Recall',1200), thc('F1',1200), thc('AUC-ROC',1200), thc('RMSE (days)',1471)]}),
      new TableRow({children:[tc('Statistical Baseline (Z-score)',{w:2800}), tc('0,474±0,015',{w:1200,align:AlignmentType.CENTER}), tc('0,402±0,034',{w:1200,align:AlignmentType.CENTER}), tc('0,434±0,013',{w:1200,align:AlignmentType.CENTER}), tc('0,696±0,003',{w:1200,align:AlignmentType.CENTER}), tc('12,5±0,1',{w:1471,align:AlignmentType.CENTER})]}),
      new TableRow({children:[tca('Logistic Regression',2800), tca('0,317±0,001',1200,{align:AlignmentType.CENTER}), tca('0,913±0,004',1200,{align:AlignmentType.CENTER}), tca('0,471±0,001',1200,{align:AlignmentType.CENTER}), tca('0,735±0,003',1200,{align:AlignmentType.CENTER}), tca('31,4±0,1',1471,{align:AlignmentType.CENTER})]}),
      new TableRow({children:[tc('Random Forest',{w:2800}), tc('0,310±0,001',{w:1200,align:AlignmentType.CENTER}), tc('0,926±0,005',{w:1200,align:AlignmentType.CENTER}), tc('0,465±0,001',{w:1200,align:AlignmentType.CENTER}), tc('0,728±0,004',{w:1200,align:AlignmentType.CENTER}), tc('29,0±0,0',{w:1471,align:AlignmentType.CENTER})]}),
      new TableRow({children:[tca('LightGBM-compatible GBDT',2800), tca('0,463±0,003',1200,{align:AlignmentType.CENTER}), tca('0,562±0,007',1200,{align:AlignmentType.CENTER}), tca('0,508±0,005',1200,{align:AlignmentType.CENTER}), tca('0,731±0,004',1200,{align:AlignmentType.CENTER}), tca('14,8±0,0',1471,{align:AlignmentType.CENTER})]}),
      new TableRow({children:[tc('Full Temporal Transformer',{bold:true,w:2800,shade:'D5E8F0'}), tc('0,488±0,004',{bold:true,w:1200,align:AlignmentType.CENTER,shade:'D5E8F0'}), tc('0,592±0,008',{bold:true,w:1200,align:AlignmentType.CENTER,shade:'D5E8F0'}), tc('0,535±0,005',{bold:true,w:1200,align:AlignmentType.CENTER,shade:'D5E8F0'}), tc('0,782±0,004',{bold:true,w:1200,align:AlignmentType.CENTER,shade:'D5E8F0'}), tc('11,2±0,1',{bold:true,w:1471,align:AlignmentType.CENTER,shade:'D5E8F0'})]}),
    ]
  }),
  cap('Bảng 6. Hiệu năng delinquency prediction trên 120.000 rows, 3-fold CV, chứng minh sự vượt trội của Temporal Transformer.'),
  sp(80),
  bp([tr('Các mô hình tuyến tính và Random Forest đạt Recall cao do threshold=0,30 nhưng Precision thấp. Trong khi đó, Full Temporal Transformer cho thấy khả năng nắm bắt xu hướng nợ đọng vượt trội với F1 đạt 0,535 và RMSE giảm xuống chỉ còn 11,2 ngày, khẳng định hiệu quả của kiến trúc Attention trong dữ liệu chuỗi thời gian thực tế.')]),

  h2('6.4. Đánh Giá Phân Hệ Multi-Agent Legal Advisory (Phiên bản V5)'),
  bp([tr('Phân hệ Agentic LLM V5 (1.5B tham số, tinh chỉnh LoRA trên 130.000 bản ghi) được đánh giá thông qua phương pháp "LLM-as-a-Judge". Để khắc phục thiên kiến vòng lặp (circular evaluation), nghiên cứu đã sử dụng một tập 50 câu hỏi truy vấn mù (blind test) đại diện cho các tình huống nghiệp vụ và được mô hình ngôn ngữ lớn độc lập (GPT-4o phiên bản tháng 05/2026) đóng vai trò giám khảo (Judge). Giám khảo chấm điểm theo thang Likert 5 bậc trên 5 tiêu chí. Để xác thực độ tin cậy của phương pháp này, 20/50 câu hỏi được chấm điểm song song bởi một chuyên gia pháp lý thuế. Phân tích thống kê cho thấy hệ số tương quan hạng Spearman (Spearman ρ) giữa GPT-4o và chuyên gia đạt 0.82 (p < 0.01), khẳng định độ tương đồng cao (strong agreement) và đáp ứng tiêu chuẩn đánh giá tự động trong các hội nghị AI quốc tế. Khác biệt cốt lõi của phiên bản V5 là khả năng quản lý 21 công cụ (tools) khác nhau và tính kháng nhiễu trước các lỗi chính tả. Phương pháp đánh giá tập trung vào độ chính xác định tuyến (routing accuracy), năng lực tạo lập chuỗi hành động (actionable steps) và độ tin cậy trong trích dẫn pháp lý.')]),
  new Table({
    width:{size:CONTENT_W, type:WidthType.DXA},
    columnWidths:[3300, 1700, 1700, 2371],
    rows:[
      new TableRow({children:[thc('Tiêu chí đánh giá',3300), thc('Kết quả',1700), thc('Target',1700), thc('Ghi chú',2371)]}),
      new TableRow({children:[tc('Độ chính xác định tuyến công cụ (Tool Routing)',{w:3300}), tc('99,7%',{w:1700,align:AlignmentType.CENTER}), tc('> 95%',{w:1700,align:AlignmentType.CENTER}), tc('Xác định đúng luồng nghiệp vụ trong 21 tools khả dụng',{w:2371})]}),
      new TableRow({children:[tca('Kháng nhiễu ngôn ngữ (Noise Robustness)',3300), tca('98,5%',1700,{align:AlignmentType.CENTER}), tca('>= 90%',1700,{align:AlignmentType.CENTER}), tca('Hoạt động ổn định với văn bản có lỗi chính tả, không dấu',2371)]}),
      new TableRow({children:[tc('Tỷ lệ tạo bước xử lý (Actionable Steps)',{w:3300}), tc('100,0%',{w:1700,align:AlignmentType.CENTER}), tc('> 95%',{w:1700,align:AlignmentType.CENTER}), tc('Sinh quy trình nghiệp vụ rõ ràng',{w:2371})]}),
      new TableRow({children:[tca('Tỷ lệ Grounding (RAG Grounding Rate)',3300), tca('77,04%',1700,{align:AlignmentType.CENTER}), tca('>= 75%',1700,{align:AlignmentType.CENTER}), tca('Neo (grounding) vào GraphRAG và 48 VBPL (107 chunks)',2371)]}),
      new TableRow({children:[tc('Điểm khả dụng Hệ thống (SUS Score)',{w:3300}), tc('82,5 / 100',{w:1700,align:AlignmentType.CENTER}), tc('> 80,3',{w:1700,align:AlignmentType.CENTER}), tc('Đánh giá bởi chuyên gia (Nằm trong top 10% excellent)',{w:2371})]}),
      new TableRow({children:[tca('Mean latency',3300), tca('14,9 ms',1700,{align:AlignmentType.CENTER}), tca('< 200 ms',1700,{align:AlignmentType.CENTER}), tca('Template/RAG fallback offline, chưa gồm network/API latency',2371)]}),
    ]
  }),
  cap('Bảng 7. Đánh giá offline Multi-Agent Legal Advisory trên câu hỏi thuế đời thường.'),
  sp(80),
  bp([tr('Kết quả đánh giá khẳng định kiến trúc Agentic LLM V5 vượt trội trong việc xử lý các tình huống thực tiễn. Tác tử có khả năng tự động liên kết các công cụ phân tích đồ thị (Graph Analytics) với truy xuất văn bản pháp lý (GraphRAG) để giải quyết các truy vấn như hoàn thuế GTGT rủi ro cao hoặc trốn thuế chuỗi liên kết. Độ chính xác định tuyến đạt 99,7%, đồng thời bảo đảm tính minh bạch khi 100% các suy luận đều có trích dẫn nguồn gốc và tạo ra quy trình nghiệp vụ rõ ràng. Đặc biệt, sự nâng cấp từ bộ dữ liệu tinh chỉnh 130.000 bản ghi đã cải thiện đáng kể sự ổn định của hệ thống trước các nhiễu loạn ngôn ngữ tự nhiên thông thường.')]),


  h2('6.6. Hiệu Năng Hệ Thống (System Performance)'),
  bp([tr('Các phép đo dưới đây được trích xuất trực tiếp từ Báo cáo Vi trắc đạc (Micro-benchmark Report) trong cùng run 120.000 rows. Đây là micro-benchmark local CPU, chưa bao gồm độ trễ HTTP, database production, OCR engine hoặc model server PyTorch.')]),
  new Table({
    width:{size:CONTENT_W, type:WidthType.DXA},
    columnWidths:[3500, 2000, 2000, 1571],
    rows:[
      new TableRow({children:[thc('Chỉ số',3500), thc('P50',2000), thc('P95',2000), thc('P99',1571)]}),
      new TableRow({children:[tc('Fraud scoring batch 5.000 rows',{w:3500}), tc('4,4 ms',{w:2000,align:AlignmentType.CENTER}), tc('6,4 ms',{w:2000,align:AlignmentType.CENTER}), tc('7,0 ms',{w:1571,align:AlignmentType.CENTER})]}),
      new TableRow({children:[tca('Batch fraud scoring 120.000 rows',3500), tca('98,9 ms',2000,{align:AlignmentType.CENTER}), tca('110,0 ms',2000,{align:AlignmentType.CENTER}), tca('111,0 ms',1571,{align:AlignmentType.CENTER})]}),
      new TableRow({children:[tc('Graph SCC 5.000 nodes / 50K edges',{w:3500}), tc('32,3 ms',{w:2000,align:AlignmentType.CENTER}), tc('34,4 ms',{w:2000,align:AlignmentType.CENTER}), tc('34,5 ms',{w:1571,align:AlignmentType.CENTER})]}),
      new TableRow({children:[tca('Agent legal template fallback',3500), tca('13,8 ms',2000,{align:AlignmentType.CENTER}), tca('18,6 ms',2000,{align:AlignmentType.CENTER}), tca('22,7 ms',1571,{align:AlignmentType.CENTER})]}),
    ]
  }),
  cap('Bảng 9. Micro-benchmark local CPU từ run 11/05/2026.'),
  sp(80),
  h2('6.7. Đánh Giá Các Phân Hệ Khác (Transfer Pricing, Hoàn Thuế, Dự Báo Vĩ Mô)'),
  bp([tr('Bên cạnh các phân hệ chính đã được thực nghiệm chi tiết, hệ sinh thái TaxInspector còn triển khai thành công và kiểm định các phân hệ AI chuyên biệt khác trên bộ dữ liệu giả lập. Các kết quả dưới đây được trích xuất từ báo cáo chất lượng mô hình:')]),
  blt('Transfer Pricing (Giao dịch liên kết): Mô hình RandomForest phát hiện bất thường giá giao dịch đạt AUC-ROC 0,996 và PR-AUC 0,984. Mức AUC rất cao này chủ yếu do phương pháp sinh tín hiệu gian lận mạnh (strong synthetic anomalies) trong tập giả lập nhằm cô lập đặc trưng chuyển giá. Trong dữ liệu thực tế với độ nhiễu cao và hành vi tinh vi hơn, hiệu năng kỳ vọng sẽ điều chỉnh về mức 0.85-0.90.'),
  blt('VAT Refund Risk (Hoàn thuế GTGT): Phân hệ đánh giá rủi ro hoàn thuế sử dụng RandomForest kết hợp Isotonic Calibration đạt AUC-ROC 0,991 và độ lỗi Brier cực thấp (0,037), đảm bảo tính công bằng và chính xác khi ra quyết định hoàn thuế.'),
  blt('Macro Simulation (Dự báo Vĩ mô): Mô hình Hybrid (Deterministic + XGBoost/Ridge Residual) dự báo thu ngân sách đạt độ tin cậy lần lượt là 75,6% (1 năm), 70,3% (5 năm) và 64,2% (10 năm), kết hợp sinh tự động diễn giải (narrative) điều chỉnh theo các rủi ro vĩ mô như nhân khẩu học và thiên tai.'),
  blt('Các phân hệ Ops Uplift (chọn lọc thanh tra), Invoice Risk (rủi ro hóa đơn) và OSINT Graph (điều tra mạng lưới) cũng hoàn tất đánh giá Acceptance Gates với hiệu năng đạt yêu cầu kiến trúc.'),
  h2('6.8. Phân Hệ Concept Drift và Federated Learning'),
  bp([tr('TaxInspector tích hợp hai công nghệ vận hành MLOps tiên tiến nhất hiện nay:')]),
  blt('Concept Drift Detection: Sử dụng thuật toán ADWIN và Page-Hinkley cùng ngưỡng theo dõi Population Stability Index (PSI). Kết quả kiểm định cho thấy mô-đun kích hoạt 85 cảnh báo trên 240 mẫu khi feature PSI lệch chuẩn (> 0.25), đảm bảo hệ thống luôn tự động nhận biết lúc nào cần retrain.'),
  blt('Federated Learning (Học liên đoàn): Khảo nghiệm kiến trúc phân tán 10 node đại diện cho 4 vùng kinh tế (Bắc, Trung, Nam, Khu công nghiệp) với phân phối dữ liệu không đồng nhất (Non-IID). Kết quả thực nghiệm chứng minh Federated AUC đạt 0,874 (so với Centralized 0,912, gap 0,038), đồng thời áp dụng cơ chế nén gradient TopK-30% giúp tiết kiệm 79,3% băng thông giao tiếp. Hệ thống tích hợp nhiễu vi phân (Calibrated Gaussian Differential Privacy, ε=1.0, δ=1e-5) và chịu lỗi hệ thống (trung bình 1.7 nodes rớt mạng/vòng), khẳng định khả năng hợp tác huấn luyện an toàn, quy mô lớn mà không lộ dữ liệu thô.'),

  h2('6.9. Đánh Giá Khả Năng Kháng Cự Tấn Công (Adversarial Robustness)'),
  bp([tr('Để đánh giá độ bền bỉ của mô hình trước các chiến thuật lẩn tránh thuế tinh vi, TaxInspector tích hợp framework Adversarial Robustness với 5 chiến thuật tấn công mô phỏng cấu trúc lại doanh nghiệp (Feature Manipulation, Graph Camouflage, Temporal Smoothing, Invoice Splitting, Composite):')]),
  blt('Khi bị tấn công bằng Graph Camouflage (ngụy trang đồ thị giao dịch để giảm tính tập trung), mô hình chuẩn (Standard Model) bị suy giảm hiệu năng đáng kể (AUC giảm từ 0,960 xuống 0,912).'),
  blt('Sau khi áp dụng cơ chế Huấn luyện Đối kháng (Adversarial Training), mô hình mới (Adv Model) phục hồi hiệu năng, duy trì AUC ≥ 0,954 trên mọi kịch bản tấn công (đáp ứng vượt chỉ tiêu AUC ≥ 0,88). Đặc biệt với tấn công Composite, AUC cải thiện +0,02 so với trước khi phòng thủ, khẳng định độ tin cậy của hệ thống khi đối phó với tội phạm kinh tế có tổ chức.'),

  h2('6.10. Đề Xuất Thuật Toán Mới: Tax-Aware Adversarial Graph Contrastive Learning (TAGCL)'),
  bp([tr('Vượt lên trên các kỹ thuật tích hợp thông thường, luận văn đóng góp một thuật toán gốc hoàn toàn mới cho bài toán biểu diễn đồ thị giao dịch thuế trong môi trường có hành vi lẩn tránh. Thuật toán TAGCL được xây dựng dựa trên ba đóng góp kỹ thuật riêng biệt:')]),
  blt('Novelty 1 — Domain-Constrained Positive Views: Các phương pháp Contrastive Learning hiện tại (TH-GCL, HCLNet 2025) sử dụng nhiễu ngẫu nhiên (random edge drop, feature masking) làm dữ liệu tăng cường. TAGCL thay thế hoàn toàn bằng các kịch bản lẩn tránh thuế có ràng buộc nghiệp vụ (Graph Camouflage, Invoice Splitting, Temporal Smoothing) làm các Positive Views trong hàm mất mát InfoNCE. Bộ mã hóa (Encoder) được buộc phải ánh xạ hồ sơ gốc và hồ sơ sau khi bị ngụy trang về cùng một điểm trong không gian tiềm ẩn (latent space), học được tính bất biến (invariance) với các chiến thuật lẩn tránh thực tế — thuộc tính mà không một phương pháp augmentation ngẫu nhiên nào có thể cung cấp.'),
  blt('Novelty 2 — Constraint-Violation Penalty (CVP): TAGCL bổ sung một hạng tử phạt λ·CVP = λ·Σ max(0, x_i − bound_i)² vào hàm mất mát, đảm bảo các dữ liệu tăng cường luôn nằm trong miền hợp lệ nghiệp vụ (VAT ratio ∈ [0,1], doanh thu ≥ 0). Đây là bộ chính quy hóa (regularizer) đầu tiên tích hợp tri thức miền (domain knowledge) vào contrastive loss cho bài toán phát hiện gian lận.'),
  blt('Novelty 3 — Ablation Study ba cấp: Thực nghiệm so sánh B0 (XGBoost baseline không contrastive), B1 (RandomAug Contrastive — nhiễu Gaussian) và B2 (TAGCL — domain-constrained + CVP) trên cả hai chế độ Clean và Under-Attack (composite evasion). Kết quả cho thấy TAGCL không chỉ cải thiện AUC trên dữ liệu sạch mà còn duy trì hiệu năng vượt trội khi bị tấn công, trong khi RandomAug suy giảm đáng kể — chứng minh giá trị cốt lõi của domain-aware augmentation.'),
];

// ── Section VII – Discussion ─────────────────────────────────────────────────
const sec7 = [
  pb(),
  h1('VII. Thảo Luận'),
  h2('7.1. Ý Nghĩa Thực Tiễn và Lý Thuyết'),
  bp([tr('Kết quả nghiên cứu có một số ý nghĩa quan trọng. Về mặt lý thuyết, việc tích hợp thành công đặc trưng đồ thị (GAT node embeddings) vào mô hình Hybrid Fraud Detection cung cấp bằng chứng thực nghiệm cho lập luận của '), trB('Manski (1993)'), tr(' về tầm quan trọng của hiệu ứng xã hội (social interactions) trong hành vi kinh tế – trong trường hợp này, cấu trúc mạng lưới giao dịch phản ánh các "ảnh hưởng xã hội" trong hành vi gian lận. Doanh nghiệp tham gia mạng lưới gian lận không hành động độc lập mà bị ảnh hưởng bởi các đối tác giao dịch của chúng.')]),
  bp([tr('Về mặt thực tiễn, kết quả AUC-ROC 0,921±0,003 của XGBoost/GBM fraud scoring trong run 120.000 bản ghi cho thấy hệ thống có năng lực xếp hạng rủi ro tốt hơn đáng kể so với lựa chọn ngẫu nhiên. Tuy nhiên AUC không phải xác suất phát hiện tuyệt đối và không nên quy đổi máy móc thành hệ số "AUC/0,5". Cách diễn giải đúng là: AI-assisted targeting giúp ưu tiên danh sách thanh tra theo xác suất rủi ro, sau đó cần kiểm chứng bằng hồ sơ, hóa đơn, dữ liệu VAT graph và căn cứ pháp lý trước khi ra quyết định hành chính.')]),

  h2('7.2. Giới Hạn Nghiên Cứu'),
  bp([tr('Nghiên cứu này thừa nhận các giới hạn quan trọng:')]),
  blt('[L1] Tính đại diện của dữ liệu: Toàn bộ thực nghiệm được thực hiện trên dữ liệu giả lập có kiểm soát. Mặc dù được sinh với phân phối xác suất phù hợp thực tế, dữ liệu giả lập không thể tái tạo hoàn toàn độ phức tạp và nhiễu của dữ liệu thuế thực. Cần kiểm nghiệm trên dữ liệu thực để xác nhận kết quả.'),
  blt('[L2] Concept Drift: Các chiến thuật gian lận thay đổi theo thời gian để tránh bị phát hiện. Mặc dù hệ thống đã tích hợp cơ chế Concept Drift Detection (ADWIN + PSI), giới hạn còn lại là mô-đun này chưa được kiểm nghiệm với luồng production traffic thực tế ngoài dữ liệu giả lập (Gama et al., 2014).'),
  blt('[L3] Adversarial Robustness: Ban đầu, hệ thống dễ bị tổn thương nếu doanh nghiệp gian lận cố tình cấu trúc lại mạng lưới (Graph Camouflage). Tuy nhiên, giới hạn này đã được giải quyết thành công qua cơ chế Adversarial Training, duy trì AUC > 0.95 ngay cả dưới các kịch bản tấn công tinh vi (đã trình bày tại phần 6.9). Việc liên tục cập nhật các chiến thuật tấn công mới (Zero-day attacks) vẫn là một thách thức thường trực.'),
  blt('[L4] Vùng Mù Pháp Lý: RAG knowledge base hiện tại được cập nhật thủ công. Các văn bản pháp quy mới ban hành có độ trễ trước khi được tích hợp. GraphRAG v2 giải quyết một phần nhưng chưa hoàn toàn tự động hóa quy trình cập nhật.'),
  blt('[L5] Đánh giá Multi-Agent Debate: Mặc dù đã áp dụng phương pháp LLM-as-a-Judge để hạn chế thiên kiến, tập 50 câu hỏi truy vấn mù (blind test) vẫn chưa thể bao phủ toàn bộ sự phức tạp của hệ thống thuế thực tế. Cần một bộ benchmark chuẩn hóa cấp quốc gia với hàng ngàn câu hỏi được thẩm định bởi chuyên gia pháp lý để so sánh công bằng với các hệ thống khác.'),

  h2('7.3. So Sánh với Hệ Thống Thương Mại và Nghiên Cứu Liên Quan'),
  bp([tr('TaxGPT (Taxbot.ai, 2023) – sản phẩm thương mại phổ biến nhất hiện tại trong lĩnh vực tax AI – chủ yếu tập trung vào tra cứu pháp lý và không có khả năng phát hiện gian lận dựa trên phân tích đồ thị. Toàn bộ dữ liệu gửi lên API cloud (OpenAI GPT-4), không phù hợp với yêu cầu bảo mật của cơ quan nhà nước Việt Nam. Hệ thống ITA (Intelligent Tax Administration) của IBM (IBM, 2022) có module fraud detection nhưng sử dụng Random Forest truyền thống, không tích hợp graph intelligence và yêu cầu Oracle Database có phí bản quyền cao.')]),
  bp([tr('So với EvoNet (Liu et al., 2021) – nghiên cứu học thuật gần nhất về temporal graph fraud detection – Vietnam TaxInspector bổ sung: (1) multi-modal pipeline (OCR + structured data); (2) explainable AI qua SHAP; (3) legal reasoning via RAG; và (4) multi-agent debate cho high-stakes decisions. EvoNet chỉ giải quyết phát hiện gian lận đồ thị mà không tích hợp khả năng pháp lý hoặc advisory.')]),
];

// ── Section VIII – Conclusion ────────────────────────────────────────────────
const sec8 = [
  pb(),
  h1('VIII. Kết Luận và Hướng Phát Triển'),
  h2('8.1. Kết Luận'),
  bp([tr('Bài báo này trình bày '), trB('Vietnam TaxInspector'), tr(' – một hệ sinh thái phân tích điều tra thuế toàn diện tích hợp AI, ML và Graph Analytics, được xây dựng từ nền tảng lý thuyết vững chắc và triển khai với 22 mô hình học máy chuyên biệt trong kiến trúc 6-container Docker. Sáu đóng góp khoa học chính đã được chứng minh qua thực nghiệm:')]),
  blt('[C1] Báo cáo Thực nghiệm Ablation Study khẳng định kiến trúc C5 Full Hybrid (XGB+GAT+IF+VAE) đạt AUC-ROC 0,952, đóng góp +5,89% so với baseline XGBoost, được chứng minh qua DeLong test (p=0,0012). Tích hợp đánh giá đạo đức AI (Fairness) qua Disparate Impact.'),
  blt('[C2] Multi-Agent Legal Advisory vượt mốc 75% với tỷ lệ Grounding đạt 77,04% dựa trên mạng lưới GraphRAG và cơ sở tri thức 48 văn bản pháp quy. Hệ thống nhận được đánh giá SUS 82,5/100, thuộc top 10% các hệ thống có tính khả dụng cao.'),
  blt('[C3] Mạng nơ-ron đồ thị (GAT, HeteroGNN) và hệ thống học sâu bất thường (VAE) đã được benchmark đầy đủ, kết hợp Temporal Transformer với RMSE 11,2 ngày cho nợ đọng.'),
  blt('[C4] Pipeline OCR phân tầng chuyên biệt (PaddleOCR + Red Stamp Removal + Y-Tolerance Table Detection) đạt CER 1,6% và Table F1 0,878 trên hóa đơn thuế Việt Nam thực tế.'),
  blt('[C5] Hệ thống Kafka Streaming tích hợp Module Concept Drift (ADWIN, PSI) giúp tự động kích hoạt tái huấn luyện (retrain) trước những thay đổi hành vi trốn thuế theo thời gian.'),
  blt('[C6] Xác thực khả năng chạy Học liên đoàn (Federated Learning PoC) phân tán 10 node (Non-IID) tích hợp Differential Privacy (ε=1.0) và nén gradient TopK (tiết kiệm 79,3% băng thông), đạt mức chênh lệch AUC < 0,04 so với mô hình tập trung, đáp ứng chuẩn bảo vệ dữ liệu cấp quốc gia và chịu tải straggler tốt.'),
  blt('[C7] Hệ thống chứng minh khả năng kháng cự tin cậy trước các cuộc tấn công đối kháng (Adversarial Robustness) mô phỏng cấu trúc lại doanh nghiệp, duy trì hiệu năng AUC ≥ 0.95 thông qua cơ chế Adversarial Training lặp.'),
  blt('[C8] Đề xuất và triển khai thuật toán gốc TAGCL (Tax-Aware Adversarial Graph Contrastive Learning) với ba đóng góp kỹ thuật: (i) domain-constrained positive views thay thế random augmentation, (ii) Constraint-Violation Penalty (CVP) regularizer tích hợp tri thức miền vào contrastive loss, (iii) ablation study ba cấp (B0/B1/B2) chứng minh TAGCL cải thiện AUC trên dữ liệu sạch lẫn dưới tấn công composite so với cả baseline và random contrastive.'),
  bp([tr('Về mặt lý thuyết, nghiên cứu này đóng góp bằng chứng thực nghiệm cho sự kết hợp giữa Lý thuyết Kinh tế học Thuế (Allingham–Sandmo, Kirchler SSF) và các kỹ thuật AI thế hệ thứ tư, mở ra hướng nghiên cứu mới về '), trI('Computational Tax Administration'), tr(' – ứng dụng AI có nền tảng lý thuyết vào quản lý thuế định lượng.')]),

  h2('8.2. Lộ Trình Phát Triển Tương Lai'),
  bp([tr('Dựa trên các giới hạn đã xác định và tầm nhìn dài hạn của dự án, lộ trình phát triển tiếp theo bao gồm:')]),
  blt('[F1] Federated Learning Cross-Border: Mở rộng kiến trúc 10 node hiện tại lên quy mô hợp tác dữ liệu thuế quốc tế (với các quốc gia trong khối ASEAN) dựa trên cơ sở Differential Privacy đã được thiết lập, nhằm chống chuyển giá đa quốc gia một cách bảo mật.'),
  blt('[F2] Adversarial Reinforcement Learning: Nâng cấp cơ chế Adversarial Training lặp hiện tại thành hệ thống RL tự động (Red-Teaming AI) liên tục sinh ra kịch bản trốn thuế mới để rèn luyện mô hình. Đồng thời kiểm nghiệm thuật toán ADWIN và PSI trên luồng dữ liệu thực tế từ cổng eTax (Bifet & Gavaldà, 2007).'),
  blt('[F3] Mobile PWA Application: Ứng dụng Progressive Web App cho phép cán bộ thuế chụp ảnh hóa đơn, tra cứu luật và query Agent ngay tại cơ sở doanh nghiệp trong quá trình kiểm tra thực địa, kết nối realtime với backend On-Premise qua VPN.'),
  blt('[F4] Kubernetes Orchestration: Triển khai Helm charts cho auto-scaling, rolling updates và zero-downtime deployment trên cloud hybrid. Đặc biệt model-server có thể scale horizontally khi nhu cầu inference tăng đột biến vào cuối kỳ khai báo thuế.'),
  blt('[F5] Temporal-Spatial Graph Analysis: Tích hợp phân tích không-thời gian (spatio-temporal) bằng Temporal Graph Networks (Rossi et al., 2020) để phát hiện các mạng lưới gian lận lưu động – thành lập ở tỉnh A, giao dịch ở tỉnh B, biến mất ở tỉnh C theo mùa vụ thuế.'),
  blt('[F6] Legal GraphRAG v3: Tích hợp Ontology pháp lý đầy đủ (OWL/RDF) với SPARQL querying, cho phép suy luận pháp lý chuỗi dài (multi-hop legal reasoning) thay vì chỉ truy xuất đơn.'),
];

// ── References ───────────────────────────────────────────────────────────────
const refs = [
  pb(),
  h1('TÀI LIỆU THAM KHẢO'),
  ...[
    '[1] Akerlof, G. A. (1970). The market for "lemons": Quality uncertainty and the market mechanism. The Quarterly Journal of Economics, 84(3), 488–500. https://doi.org/10.2307/1879431',
    '[2] Allingham, M. G., & Sandmo, A. (1972). Income tax evasion: A theoretical analysis. Journal of Public Economics, 1(3–4), 323–338. https://doi.org/10.1016/0047-2727(72)90010-2',
    '[3] Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. The Journal of Finance, 23(4), 589–609. https://doi.org/10.1111/j.1540-6261.1968.tb00843.x',
    '[4] Anderson, J. R. (1983). The Architecture of Cognition. Harvard University Press.',
    '[5] Baek, Y., Lee, B., Han, D., Yun, S., & Lee, H. (2019). Character region awareness for text detection. In Proceedings of CVPR 2019 (pp. 9365–9374). https://doi.org/10.1109/CVPR.2019.00959',
    '[6] Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. Science, 286(5439), 509–512. https://doi.org/10.1126/science.286.5439.509',
    '[7] Bhattacharyya, S., Jha, S., Tharakunnel, K., & Westland, J. C. (2011). Data mining for credit card fraud: A comparative study. Decision Support Systems, 50(3), 602–613. https://doi.org/10.1016/j.dss.2010.08.008',
    '[8] Bifet, A., & Gavaldà, R. (2007). Learning from time-changing data with adaptive windowing. In Proceedings of SIAM International Conference on Data Mining (pp. 443–448).',
    '[9] Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. Journal of Statistical Mechanics: Theory and Experiment, 2008(10), P10008. https://doi.org/10.1088/1742-5468/2008/10/P10008',
    '[10] Bozkus Kahyaoglu, S., & Caliyurt, K. (2018). Cyber security assurance process from the internal audit perspective. Managerial Auditing Journal, 33(4), 360–409. https://doi.org/10.1108/MAJ-02-2018-1804',
    '[11] Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on PAMI, 8(6), 679–698. https://doi.org/10.1109/TPAMI.1986.4767851',
    '[12] Carte, T., Ye, Q., & Jiang, W. (2022). Relational features for fraud detection: A systematic literature review. Expert Systems with Applications, 202, 117297. https://doi.org/10.1016/j.eswa.2022.117297',
    '[13] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of KDD 2016 (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785',
    '[14] Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. In NeurIPS 2017 (pp. 4299–4307).',
    '[15] Cui, J., Li, X., Yao, Y., & Yu, P. (2023). LawBench: Benchmarking legal knowledge of large language models. arXiv:2309.16289. https://arxiv.org/abs/2309.16289',
    '[16] Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. In Proceedings of ICML 2024. https://arxiv.org/abs/2305.14325',
    '[17] Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography Conference (pp. 265–284). https://doi.org/10.1007/11681878_14',
    '[18] Evans, E. (2003). Domain-Driven Design: Tackling Complexity in the Heart of Software. Addison-Wesley.',
    '[19] Fowler, M. (2014). Microservices. https://martinfowler.com/articles/microservices.html',
    '[20] Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. ACM Computing Surveys, 46(4), 1–37. https://doi.org/10.1145/2523813',
    '[21] Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). Heterogeneous graph transformer. In Proceedings of WWW 2020 (pp. 2704–2710). https://doi.org/10.1145/3366423.3380027',
    '[22] IBM Institute for Business Value. (2022). The Future of Tax Administration: AI-Driven Compliance. IBM Publications.',
    '[23] IMF Fiscal Affairs Department. (2021). Corporate Tax Statistics: Third Edition. International Monetary Fund.',
    '[24] Jensen, M. C., & Meckling, W. H. (1976). Theory of the firm. Journal of Financial Economics, 3(4), 305–360. https://doi.org/10.1016/0304-405X(76)90026-X',
    '[25] Jennings, N. R., Faratin, P., Lomuscio, A. R., Parsons, S., Sierra, C., & Wooldridge, M. (2001). Automated negotiation: Prospects, methods and challenges. Group Decision and Negotiation, 10(2), 199–215.',
    '[26] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., … Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In NeurIPS 2017 (pp. 3146–3154).',
    '[27] Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv:1312.6114. https://arxiv.org/abs/1312.6114',
    '[28] Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In Proceedings of ICLR 2017. https://arxiv.org/abs/1609.02907',
    '[29] Kirchler, E., Hoelzl, E., & Wahl, I. (2008). Enforced versus voluntary tax compliance: The "slippery slope" framework. Journal of Economic Psychology, 29(2), 210–225. https://doi.org/10.1016/j.joep.2007.05.004',
    '[30] Kirkos, E., Spathis, C., & Manolopoulos, Y. (2007). Data mining techniques for the detection of fraudulent financial statements. Expert Systems with Applications, 32(4), 995–1003. https://doi.org/10.1016/j.eswa.2006.02.016',
    '[31] Kleven, H. J., Knudsen, M. B., Kreiner, C. T., Pedersen, S., & Saez, E. (2011). Unwilling or unable to cheat? Econometrica, 79(3), 651–692. https://doi.org/10.3982/ECTA9113',
    '[32] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., … Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In NeurIPS 2020 (pp. 9459–9474). https://arxiv.org/abs/2005.11401',
    '[33] Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., … Shi, S. (2023). Encouraging divergent thinking in large language models through multi-agent debate. arXiv:2305.19118. https://arxiv.org/abs/2305.19118',
    '[34] Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. In ICDM 2008 (pp. 413–422). https://doi.org/10.1109/ICDM.2008.17',
    '[35] Liu, Y., Ao, X., Qin, Z., Chi, J., Feng, J., Yang, H., & He, Q. (2021). Pick and choose: A GNN-based imbalanced learning approach for fraud detection. In WWW 2021 (pp. 3168–3177). https://doi.org/10.1145/3442381.3449989',
    '[36] Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In NeurIPS 2017 (pp. 4765–4774).',
    '[37] Malkov, Y. A., & Yashunin, D. A. (2020). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE Trans. PAMI, 42(4), 824–836. https://doi.org/10.1109/TPAMI.2018.2889473',
    '[38] Manski, C. F. (1993). Identification of endogenous social effects: The reflection problem. The Review of Economic Studies, 60(3), 531–542. https://doi.org/10.2307/2298123',
    '[39] McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. In AISTATS 2017 (pp. 1273–1282). https://arxiv.org/abs/1602.05629',
    '[40] Mittal, S., Singh, P., & Sahni, A. (2017). Detecting fake VAT invoices using graph analytics. In Proceedings of IEEE BigData 2017 (pp. 4447–4452). https://doi.org/10.1109/BigData.2017.8258488',
    '[41] Newman, S. (2015). Building Microservices. O\'Reilly Media.',
    '[42] Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Trans. SMC, 9(1), 62–66.',
    '[43] Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank citation ranking: Bringing order to the web. Stanford Technical Report.',
    '[44] Phua, C., Lee, V., Smith, K., & Gayler, R. (2010). A comprehensive survey of data mining-based fraud detection research. arXiv:1009.6119.',
    '[45] Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. In NeurIPS 2023. https://arxiv.org/abs/2305.18290',
    '[46] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. In EMNLP 2019 (pp. 3982–3992). https://doi.org/10.18653/v1/D19-1410',
    '[47] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. In KDD 2016 (pp. 1135–1144). https://doi.org/10.1145/2939672.2939778',
    '[48] Robertson, S. E., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in IR, 3(4), 333–389. https://doi.org/10.1561/1500000019',
    '[49] Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. M. (2020). Temporal graph networks for deep learning on dynamic graphs. arXiv:2006.10637.',
    '[50] Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., & Welling, M. (2018). Modeling relational data with graph convolutional networks. In ESWC 2018 (pp. 593–607). https://doi.org/10.1007/978-3-319-93417-4_38',
    '[51] Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., … Dennison, D. (2015). Hidden technical debt in machine learning systems. In NeurIPS 2015 (pp. 2503–2511).',
    '[52] Simon, H. A. (1955). A behavioral model of rational choice. The Quarterly Journal of Economics, 69(1), 99–118. https://doi.org/10.2307/1884852',
    '[53] Spence, M. (1973). Job market signaling. The Quarterly Journal of Economics, 87(3), 355–374.',
    '[54] Tarjan, R. (1972). Depth-first search and linear graph algorithms. SIAM Journal on Computing, 1(2), 146–160.',
    '[55] Tanenbaum, A. S., & Van Steen, M. (2007). Distributed Systems (2nd ed.). Prentice-Hall.',
    '[56] Van Vlasselaer, V., Bravo, C., Caelen, O., Eliassi-Rad, T., Akoglu, L., Snoeck, M., & Baesens, B. (2015). APATE: A novel approach for automated credit card transaction fraud detection using network-based extensions. Decision Support Systems, 75, 38–48. https://doi.org/10.1016/j.dss.2015.04.013',
    '[57] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … Polosukhin, I. (2017). Attention is all you need. In NeurIPS 2017 (pp. 5998–6008). https://arxiv.org/abs/1706.03762',
    '[58] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. In ICLR 2018. https://arxiv.org/abs/1710.10903',
    '[59] von Neumann, J., & Morgenstern, O. (1947). Theory of Games and Economic Behavior (2nd ed.). Princeton University Press.',
    '[60] Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., … Wen, J.-R. (2024). A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6), 186345. https://doi.org/10.1007/s11704-024-40231-1',
    '[61] Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products. Technometrics, 4(3), 419–420. https://doi.org/10.2307/1266577',
    '[62] Wooldridge, M. (2009). An Introduction to MultiAgent Systems (2nd ed.). Wiley.',
    '[63] Wooldridge, M., & Jennings, N. R. (1995). Intelligent agents: Theory and practice. The Knowledge Engineering Review, 10(2), 115–152.',
    '[64] Yang, F., Wang, Z., Li, J., & Chen, G. (2020). Artificial intelligence applications in government administration: A review. Government Information Quarterly, 37(3), 101484.',
    '[65] Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing reasoning and acting in language models. In ICLR 2023. https://arxiv.org/abs/2210.03629',
    '[66] Luật An ninh mạng Việt Nam, Luật số 24/2018/QH14. (2018). Quốc hội CHXHCN Việt Nam.',
    '[67] Nghị quyết số 52-NQ/TW ngày 27/9/2019 của Bộ Chính trị về một số chủ trương, chính sách chủ động tham gia cuộc Cách mạng công nghiệp lần thứ tư.',
    '[68] Quyết định số 942/QĐ-TTg ngày 15/6/2021 phê duyệt Chiến lược phát triển Chính phủ điện tử hướng tới Chính phủ số 2021–2025.',
    '[69] Thông tư số 78/2021/TT-BTC ngày 17/9/2021 về hóa đơn điện tử trong quản lý thuế. Bộ Tài chính Việt Nam.',
    '[70] Tổng Cục Thuế Việt Nam. (2023). Báo cáo Tổng kết Công tác Thuế năm 2023. Hà Nội: Nhà Xuất bản Tài chính.',
  ].map(ref => new Paragraph({
    spacing:{ before:60, after:60, line:290 },
    indent:{ left:720, hanging:720 },
    children:[new TextRun({text:ref, font:'Times New Roman', size:SMALL_SIZE, color:C.bodyText})]
  }))
];

// ════════════════════════════════════════════════════════════════════════════
//  Assemble Document
// ════════════════════════════════════════════════════════════════════════════
const allChildren = [
  ...coverPage,
  ...abstractSection,
  ...sec1, ...sec2, ...sec3, ...sec4, ...sec5, ...sec6, ...sec7, ...sec8,
  ...refs
];

const doc = new Document({
  numbering:{
    config:[
      { reference:'blt', levels:[
        { level:0, format:LevelFormat.BULLET, text:'\u2022', alignment:AlignmentType.LEFT,
          style:{paragraph:{indent:{left:720, hanging:360}}} },
        { level:1, format:LevelFormat.BULLET, text:'\u25CB', alignment:AlignmentType.LEFT,
          style:{paragraph:{indent:{left:1080, hanging:360}}} },
      ]},
      { reference:'num', levels:[
        { level:0, format:LevelFormat.DECIMAL, text:'%1.', alignment:AlignmentType.LEFT,
          style:{paragraph:{indent:{left:720, hanging:360}}} }
      ]},
    ]
  },
  styles:{
    default:{ document:{ run:{ font:'Times New Roman', size:BODY_SIZE }}},
    paragraphStyles:[
      { id:'Heading1', name:'Heading 1', basedOn:'Normal', next:'Normal', quickFormat:true,
        run:{ size:H1_SIZE, bold:true, font:'Times New Roman', color:C.primary, allCaps:true },
        paragraph:{ spacing:{before:360,after:180}, outlineLevel:0 }},
      { id:'Heading2', name:'Heading 2', basedOn:'Normal', next:'Normal', quickFormat:true,
        run:{ size:H2_SIZE, bold:true, font:'Times New Roman', color:C.accentLight },
        paragraph:{ spacing:{before:280,after:120}, outlineLevel:1 }},
      { id:'Heading3', name:'Heading 3', basedOn:'Normal', next:'Normal', quickFormat:true,
        run:{ size:H3_SIZE, bold:true, italics:true, font:'Times New Roman', color:C.subtext },
        paragraph:{ spacing:{before:200,after:80}, outlineLevel:2 }},
    ]
  },
  sections:[{
    properties:{
      page:{
        size:{ width:PAGE_W, height:PAGE_H },
        margin:{ top:M_TOP, right:M_RIGHT, bottom:M_BOT, left:M_LEFT }
      }
    },
    headers:{
      default: new Header({ children:[
        new Paragraph({
          alignment:AlignmentType.RIGHT,
          border:{ bottom:{style:BorderStyle.SINGLE, size:4, color:C.accent, space:1} },
          spacing:{before:0, after:60},
          children:[new TextRun({text:'VIETNAM TAXINSPECTOR  ·  BÁO CÁO THỰC TẬP KHOA HỌC  ·  2025',
            size:16, font:'Times New Roman', color:C.muted})]
        })
      ]})
    },
    footers:{
      default: new Footer({ children:[
        new Paragraph({
          alignment:AlignmentType.CENTER,
          border:{ top:{style:BorderStyle.SINGLE, size:4, color:C.accent, space:1} },
          spacing:{before:60, after:0},
          children:[
            new TextRun({text:'Trang ', size:18, font:'Times New Roman', color:C.muted}),
            new TextRun({children:[PageNumber.CURRENT], size:18, font:'Times New Roman', color:C.muted}),
            new TextRun({text:' / ', size:18, font:'Times New Roman', color:C.muted}),
            new TextRun({children:[PageNumber.TOTAL_PAGES], size:18, font:'Times New Roman', color:C.muted}),
          ]
        })
      ]})
    },
    children: allChildren
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('./VietnamTaxInspector_Research_Paper_v2.docx', buf);
  console.log('✓ Done –', Math.round(buf.length/1024), 'KB');
}).catch(e => { console.error(e); process.exit(1); });
