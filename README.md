# FastAPI Medical Speech-to-Text API

This is a simple FastAPI project that allows you to upload audio files and transcribe them using **Google Cloud Speech-to-Text**.  

---

## Prerequisites

- Python 3.11+ installed
- Google Cloud project with:
  - Service Account JSON key
  - Permissions: **Cloud Storage Admin**, **Speech-to-Text Admin**
  - GCS bucket created to store audio files
- `pip` package manager

---

## Setup Instructions

1. **Clone the repository**

```bash
git clone <repository_url>
cd <repository_folder>
