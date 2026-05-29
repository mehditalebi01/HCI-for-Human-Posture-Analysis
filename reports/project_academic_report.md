# گزارش علمی کوتاه پروژه HCI for Human Posture Analysis

## 1. هدف پروژه

این پروژه یک pipeline ماژولار برای تحلیل وضعیت بدن انسان در ویدیوهای ورزشی است که از ورودی ویدیو شروع می‌شود و تا استخراج featureهای بیومکانیکی دوبعدی ادامه پیدا می‌کند [OpenCV Video I/O](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html), [Uhlrich et al., 2023](https://pubmed.ncbi.nlm.nih.gov/37856442/).  
هدف فعلی پروژه تولید یک سیستم آموزشی و قابل توسعه است، نه یک سامانه کامل بالینی یا معادل کامل OpenCap [Uhlrich et al., 2023](https://pubmed.ncbi.nlm.nih.gov/37856442/), [Pagnon et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754/).

## 2. معماری کلی

معماری پروژه به صورت لایه‌ای طراحی شده است تا هر مرحله مانند ورودی ویدیو، تخمین pose، smoothing و استخراج feature به صورت جداگانه تست و توسعه داده شود [Gamma et al., 1994](https://www.oreilly.com/library/view/design-patterns-elements/0201633612/), [OpenCV Video I/O](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html).  
کدهای reusable داخل `src/hpa/` قرار دارند و scriptهای اجرایی داخل `src/scripts/` هستند تا منطق علمی از رابط command-line جدا بماند [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

```text
Video Input
  -> Frame Extraction
  -> 2D Human Pose Estimation
  -> Pose Smoothing and Quality Check
  -> OpenCap-inspired 2D Biomechanical Features
  -> Future: Tracking, 3D Pose, Action Recognition, Risk Scoring
```

## 3. Stage 1: استخراج فریم از ویدیو

در Stage 1 ویدیو با OpenCV خوانده می‌شود و هر `N` فریم یک تصویر ذخیره می‌شود، چون `cv2.VideoCapture` روش استاندارد OpenCV برای خواندن frame-by-frame ویدیو است [OpenCV VideoCapture](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html).  
پارامتر پیش‌فرض `step=30` برای کاهش حجم داده و ساخت یک baseline سبک انتخاب شده است، زیرا در پردازش ویدیویی کاهش نرخ نمونه‌برداری یک روش عملی برای کم‌کردن هزینه محاسباتی است [OpenCV Video I/O](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html).

## 4. Stage 2: تخمین pose دوبعدی

در Stage 2 از RTMLib استفاده شده است، چون RTMLib یک wrapper سبک برای اجرای مدل‌های RTMPose و detectorهای مرتبط با نصب ساده‌تر نسبت به راه‌اندازی کامل MMPose است [RTMLib GitHub](https://github.com/Tau-J/rtmlib), [MMPose Documentation](https://mmpose.readthedocs.io/).  
مدل RTMPose برای تخمین keypointهای دوبعدی انتخاب شده است، چون در خانواده مدل‌های top-down pose estimation برای real-time multi-person pose estimation طراحی شده است [Jiang et al., RTMPose](https://arxiv.org/abs/2303.07399).  
تشخیص انسان در این مرحله داخل `rtmlib.Body` انجام می‌شود و RTMLib برای حالت body از YOLOX به عنوان detector استفاده می‌کند [RTMLib GitHub](https://github.com/Tau-J/rtmlib), [Ge et al., 2021](https://arxiv.org/abs/2107.08430).  
خروجی این مرحله CSV شامل `frame_name`, `person_id`, `keypoint_id`, `x`, `y`, و `confidence` است، و این نوع نمایش row-based برای keypointها برای تحلیل آماری و پردازش جدولی مناسب است [pandas Documentation](https://pandas.pydata.org/docs/).

## 5. اجرای GPU و ONNX Runtime

برای اجرای سریع‌تر مدل‌ها از backend `onnxruntime` و در صورت وجود CUDA از `CUDAExecutionProvider` استفاده می‌شود، چون ONNX Runtime اجرای مدل‌های ONNX را روی CPU و GPU پشتیبانی می‌کند [ONNX Runtime CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).  
پارامتر `mode="lightweight"` در live demo برای افزایش FPS انتخاب شده است، زیرا مدل‌های سبک‌تر معمولا دقت کمتر ولی سرعت بالاتر دارند و این trade-off در کاربردهای real-time رایج است [RTMLib GitHub](https://github.com/Tau-J/rtmlib), [Jiang et al., RTMPose](https://arxiv.org/abs/2303.07399).

## 6. Stage 3: smoothing و کنترل کیفیت pose

در Stage 3 مختصات `x` و `y` برای هر `person_id` و `keypoint_id` با moving average هموار می‌شوند، چون moving average یک روش ساده و قابل توضیح برای کاهش jitter در سیگنال‌های زمانی است [pandas Rolling Window](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html).  
مقدار `confidence` تغییر داده نمی‌شود تا score اصلی مدل حفظ شود و مراحل بعدی بتوانند کیفیت keypointها را مستقل از smoothing بررسی کنند [Jiang et al., RTMPose](https://arxiv.org/abs/2303.07399).  
گزارش کیفیت شامل تعداد کل keypointها، میانگین confidence، تعداد keypointهای کم‌اعتماد، نسبت کم‌اعتمادی و تعداد frameهای پردازش‌شده است تا داده قبل از biomechanics قابل ارزیابی باشد [pandas Documentation](https://pandas.pydata.org/docs/).

## 7. Stage 4: OpenCap-style biomechanical features

در Stage 4 featureهای بیومکانیکی دوبعدی مانند زاویه زانو، زاویه لگن، lean تنه و عدم‌تقارن چپ و راست از keypointهای smoothed استخراج می‌شوند [Uhlrich et al., 2023](https://pubmed.ncbi.nlm.nih.gov/37856442/), [Pagnon et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754/).  
این featureها فقط OpenCap-inspired هستند، چون OpenCap کامل از workflow ضبط مشخص و بازسازی سه‌بعدی برای محاسبه kinematics و dynamics استفاده می‌کند [Uhlrich et al., 2023](https://pubmed.ncbi.nlm.nih.gov/37856442/).  
زاویه‌های فعلی از keypointهای دوبعدی در صفحه تصویر محاسبه می‌شوند و نباید معادل زاویه مفصل سه‌بعدی یا اندازه‌گیری بالینی فرض شوند [Drazan et al., 2022](https://www.mdpi.com/1424-8220/22/5/1729).  
خروجی plotها با Matplotlib ذخیره می‌شود تا روند زاویه‌ها و asymmetry در طول فریم‌ها به صورت قابل مشاهده بررسی شود [Matplotlib savefig](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html).

## 8. محدودیت‌های فعلی

پروژه فعلا tracking واقعی ندارد، بنابراین `person_id` فقط شماره فرد داخل همان فریم است و هویت پایدار فرد در طول ویدیو را تضمین نمی‌کند [RTMLib GitHub](https://github.com/Tau-J/rtmlib).  
خروجی detection به صورت جدا ذخیره نمی‌شود، چون detector فعلا داخل RTMLib اجرا می‌شود و فقط خروجی نهایی pose در pipeline ذخیره می‌شود [RTMLib GitHub](https://github.com/Tau-J/rtmlib), [Ge et al., 2021](https://arxiv.org/abs/2107.08430).  
برای رسیدن به تحلیل بیومکانیکی معتبرتر، پروژه در آینده به tracking، داده چنددوربینه calibrated یا integration با OpenCap/Pose2Sim نیاز دارد [Uhlrich et al., 2023](https://pubmed.ncbi.nlm.nih.gov/37856442/), [Pagnon et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754/).

## 9. جمع‌بندی

تا این مرحله پروژه یک pipeline علمی و قابل توسعه برای استخراج pose دوبعدی، هموارسازی keypointها و تولید featureهای بیومکانیکی surrogate دارد [Jiang et al., RTMPose](https://arxiv.org/abs/2303.07399), [Uhlrich et al., 2023](https://pubmed.ncbi.nlm.nih.gov/37856442/).  
استفاده از RTMLib، ONNX Runtime، OpenCV، pandas و Matplotlib باعث شده است که پیاده‌سازی ساده، قابل تکرار و مناسب برای توسعه دانشگاهی باقی بماند [RTMLib GitHub](https://github.com/Tau-J/rtmlib), [ONNX Runtime CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), [OpenCV Video I/O](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html), [pandas Documentation](https://pandas.pydata.org/docs/), [Matplotlib Documentation](https://matplotlib.org/stable/).

## References

1. Uhlrich, S. D. et al. OpenCap: Human movement dynamics from smartphone videos. Nature Biomedical Engineering, 2023. https://pubmed.ncbi.nlm.nih.gov/37856442/  
2. Pagnon, D., Domalain, M., & Reveret, L. Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics. Sensors, 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754/  
3. Jiang, T. et al. RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose. arXiv, 2023. https://arxiv.org/abs/2303.07399  
4. Ge, Z. et al. YOLOX: Exceeding YOLO Series in 2021. arXiv, 2021. https://arxiv.org/abs/2107.08430  
5. RTMLib GitHub repository. https://github.com/Tau-J/rtmlib  
6. ONNX Runtime CUDA Execution Provider documentation. https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html  
7. OpenCV video display and capture documentation. https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html  
8. pandas rolling window documentation. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html  
9. Matplotlib savefig documentation. https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html  
10. Drazan, J. F. et al. Accuracy Assessment of Joint Angles Estimated from 2D and 3D Camera Measurements. Sensors, 2022. https://www.mdpi.com/1424-8220/22/5/1729
