# استخدم صورة Python الأساسية
FROM python:3.11-slim

# تثبيت الأدوات اللازمة لبناء الحزم
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libssl-dev \
    libffi-dev \
    && apt-get clean

# تعيين دليل العمل
WORKDIR /app

# نسخ متطلبات المشروع وتثبيتها
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع
COPY . .

# تعيين الأمر الافتراضي لتشغيل البوت
CMD ["python", "bot.py"]
