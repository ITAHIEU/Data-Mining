# EDA - AI Job Market Dataset

## 1) Tổng quan dữ liệu
- Dataset: ai_job_dataset_merged_cleaned.csv
- Số dòng: 30,000
- Số cột: 26

## 2) Kiểu dữ liệu và giá trị thiếu
| column | dtype | missing | missing_pct |
| --- | --- | --- | --- |
| application_deadline | object | 0 | 0.0000 |
| benefits_score | float64 | 0 | 0.0000 |
| company_location | object | 0 | 0.0000 |
| company_name | object | 0 | 0.0000 |
| company_size | object | 0 | 0.0000 |
| days_to_deadline | int64 | 0 | 0.0000 |
| education_required | object | 0 | 0.0000 |
| education_required_ord | int64 | 0 | 0.0000 |
| employee_residence | object | 0 | 0.0000 |
| employment_type | object | 0 | 0.0000 |
| experience_level | object | 0 | 0.0000 |
| experience_level_ord | int64 | 0 | 0.0000 |
| home_country_match | int64 | 0 | 0.0000 |
| industry | object | 0 | 0.0000 |
| job_description_length | int64 | 0 | 0.0000 |
| job_id | object | 0 | 0.0000 |
| job_title | object | 0 | 0.0000 |
| posting_date | object | 0 | 0.0000 |
| remote_ratio | int64 | 0 | 0.0000 |
| required_skills | object | 0 | 0.0000 |
| salary_currency | object | 0 | 0.0000 |
| salary_local | float64 | 0 | 0.0000 |
| salary_usd | float64 | 0 | 0.0000 |
| skills_count | int64 | 0 | 0.0000 |
| source_file | object | 0 | 0.0000 |
| years_experience | int64 | 0 | 0.0000 |

- Tổng giá trị thiếu: 0

## 3) Kiểm tra trùng lặp
- Số dòng trùng lặp toàn bộ: 0
- Số job_id trùng lặp: 15000
- Số cặp (job_id, source_file) trùng lặp: 0

## 4) Thống kê mô tả các biến số
| feature | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benefits_score | 30000.0000 | 7.5019 | 1.4475 | 5.0000 | 6.3000 | 7.5000 | 8.8000 | 10.0000 |
| days_to_deadline | 30000.0000 | 43.5005 | 17.2830 | 13.0000 | 29.0000 | 43.0000 | 59.0000 | 74.0000 |
| education_required_ord | 30000.0000 | 3.4947 | 1.1154 | 2.0000 | 3.0000 | 3.0000 | 4.0000 | 5.0000 |
| experience_level_ord | 30000.0000 | 2.5106 | 1.1172 | 1.0000 | 2.0000 | 3.0000 | 4.0000 | 4.0000 |
| home_country_match | 30000.0000 | 0.7071 | 0.4551 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| job_description_length | 30000.0000 | 1502.0837 | 575.4180 | 500.0000 | 1001.0000 | 1512.0000 | 1997.0000 | 2499.0000 |
| remote_ratio | 30000.0000 | 49.8400 | 40.8293 | 0.0000 | 0.0000 | 50.0000 | 100.0000 | 100.0000 |
| salary_local | 30000.0000 | 121862.5638 | 66872.8784 | 16621.0000 | 71946.2500 | 103944.5000 | 155791.7500 | 307573.2500 |
| salary_usd | 30000.0000 | 117623.4489 | 59145.0400 | 16621.0000 | 72575.7500 | 103206.5000 | 150921.7500 | 276912.8750 |
| skills_count | 30000.0000 | 3.9927 | 0.8144 | 3.0000 | 3.0000 | 4.0000 | 5.0000 | 5.0000 |
| years_experience | 30000.0000 | 6.3094 | 5.5724 | 0.0000 | 2.0000 | 5.0000 | 10.0000 | 19.0000 |

## 5) Phân tích nhóm biến phân loại
### job_title - Top 10
| job_title | count | pct |
| --- | --- | --- |
| Machine Learning Engineer | 1596 | 5.3200 |
| Machine Learning Researcher | 1542 | 5.1400 |
| Autonomous Systems Engineer | 1532 | 5.1100 |
| AI Architect | 1529 | 5.1000 |
| Robotics Engineer | 1521 | 5.0700 |
| Data Engineer | 1518 | 5.0600 |
| AI Software Engineer | 1514 | 5.0500 |
| AI Product Manager | 1507 | 5.0200 |
| Computer Vision Engineer | 1504 | 5.0100 |
| Deep Learning Engineer | 1504 | 5.0100 |

### experience_level - Top 10
| experience_level | count | pct |
| --- | --- | --- |
| EX | 7603 | 25.3400 |
| MI | 7545 | 25.1500 |
| SE | 7482 | 24.9400 |
| EN | 7370 | 24.5700 |

### employment_type - Top 10
| employment_type | count | pct |
| --- | --- | --- |
| CT | 7562 | 25.2100 |
| FT | 7509 | 25.0300 |
| PT | 7466 | 24.8900 |
| FL | 7463 | 24.8800 |

### company_size - Top 10
| company_size | count | pct |
| --- | --- | --- |
| L | 10085 | 33.6200 |
| S | 9982 | 33.2700 |
| M | 9933 | 33.1100 |

### industry - Top 10
| industry | count | pct |
| --- | --- | --- |
| Consulting | 2041 | 6.8000 |
| Retail | 2041 | 6.8000 |
| Automotive | 2035 | 6.7800 |
| Government | 2033 | 6.7800 |
| Technology | 2022 | 6.7400 |
| Media | 2017 | 6.7200 |
| Real Estate | 2012 | 6.7100 |
| Finance | 2002 | 6.6700 |
| Manufacturing | 1997 | 6.6600 |
| Telecommunications | 1992 | 6.6400 |

### company_location - Top 10
| company_location | count | pct |
| --- | --- | --- |
| Switzerland | 1565 | 5.2200 |
| Germany | 1560 | 5.2000 |
| Canada | 1550 | 5.1700 |
| Denmark | 1537 | 5.1200 |
| Singapore | 1526 | 5.0900 |
| China | 1523 | 5.0800 |
| France | 1518 | 5.0600 |
| Israel | 1509 | 5.0300 |
| United Kingdom | 1509 | 5.0300 |
| Ireland | 1507 | 5.0200 |

### salary_currency - Top 10
| salary_currency | count | pct |
| --- | --- | --- |
| USD | 19410 | 64.7000 |
| EUR | 5256 | 17.5200 |
| GBP | 1509 | 5.0300 |
| CHF | 819 | 2.7300 |
| CAD | 781 | 2.6000 |
| SGD | 762 | 2.5400 |
| JPY | 742 | 2.4700 |
| AUD | 721 | 2.4000 |

### education_required - Top 10
| education_required | count | pct |
| --- | --- | --- |
| Bachelor | 7652 | 25.5100 |
| Associate | 7473 | 24.9100 |
| PhD | 7439 | 24.8000 |
| Master | 7436 | 24.7900 |

### source_file - Top 10
| source_file | count | pct |
| --- | --- | --- |
| ai_job_dataset.csv | 15000 | 50.0000 |
| ai_job_dataset1.csv | 15000 | 50.0000 |

## 6) Phân bố một số biến quan trọng
| feature | mean | median | std | p25 | p75 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| salary_usd | 117623.4489 | 103206.5000 | 59145.0400 | 72575.7500 | 150921.7500 | 16621.0000 | 276912.8750 |
| salary_local | 121862.5638 | 103944.5000 | 66872.8784 | 71946.2500 | 155791.7500 | 16621.0000 | 307573.2500 |
| years_experience | 6.3094 | 5.0000 | 5.5724 | 2.0000 | 10.0000 | 0.0000 | 19.0000 |
| remote_ratio | 49.8400 | 50.0000 | 40.8293 | 0.0000 | 100.0000 | 0.0000 | 100.0000 |
| benefits_score | 7.5019 | 7.5000 | 1.4475 | 6.3000 | 8.8000 | 5.0000 | 10.0000 |
| skills_count | 3.9927 | 4.0000 | 0.8144 | 3.0000 | 5.0000 | 3.0000 | 5.0000 |
| days_to_deadline | 43.5005 | 43.0000 | 17.2830 | 29.0000 | 59.0000 | 13.0000 | 74.0000 |
| job_description_length | 1502.0837 | 1512.0000 | 575.4180 | 1001.0000 | 1997.0000 | 500.0000 | 2499.0000 |

## 7) Lương theo cấp độ kinh nghiệm
| experience_level | count | mean | median | min | max |
| --- | --- | --- | --- | --- | --- |
| EX | 7603 | 189031.8946 | 188028.0000 | 43491.0000 | 276912.8750 |
| SE | 7482 | 125049.3157 | 123311.0000 | 32792.0000 | 247728.0000 |
| MI | 7545 | 89766.3893 | 88565.0000 | 23010.0000 | 179375.0000 |
| EN | 7370 | 64937.2570 | 63799.0000 | 16621.0000 | 131006.0000 |

## 8) Tương quan giữa các biến số
| feature | salary_usd | salary_local | years_experience | remote_ratio | benefits_score | skills_count | days_to_deadline | job_description_length | home_country_match | experience_level_ord | education_required_ord |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| salary_usd | 1.0000 | 0.8772 | 0.7530 | 0.0088 | -0.0017 | -0.0052 | 0.0004 | -0.0109 | -0.0012 | 0.7717 | -0.0075 |
| salary_local | 0.8772 | 1.0000 | 0.6510 | 0.0103 | -0.0073 | -0.0032 | -0.0009 | -0.0088 | -0.0022 | 0.6671 | -0.0102 |
| years_experience | 0.7530 | 0.6510 | 1.0000 | 0.0095 | -0.0022 | -0.0070 | -0.0032 | -0.0120 | 0.0036 | 0.9258 | -0.0089 |
| remote_ratio | 0.0088 | 0.0103 | 0.0095 | 1.0000 | 0.0015 | -0.0012 | -0.0080 | 0.0056 | -0.0000 | 0.0095 | -0.0052 |
| benefits_score | -0.0017 | -0.0073 | -0.0022 | 0.0015 | 1.0000 | 0.0072 | -0.0030 | 0.0019 | 0.0074 | 0.0007 | 0.0003 |
| skills_count | -0.0052 | -0.0032 | -0.0070 | -0.0012 | 0.0072 | 1.0000 | -0.0112 | 0.0111 | 0.0103 | -0.0037 | -0.0043 |
| days_to_deadline | 0.0004 | -0.0009 | -0.0032 | -0.0080 | -0.0030 | -0.0112 | 1.0000 | 0.0081 | 0.0001 | 0.0007 | 0.0028 |
| job_description_length | -0.0109 | -0.0088 | -0.0120 | 0.0056 | 0.0019 | 0.0111 | 0.0081 | 1.0000 | -0.0063 | -0.0113 | 0.0013 |
| home_country_match | -0.0012 | -0.0022 | 0.0036 | -0.0000 | 0.0074 | 0.0103 | 0.0001 | -0.0063 | 1.0000 | 0.0016 | -0.0030 |
| experience_level_ord | 0.7717 | 0.6671 | 0.9258 | 0.0095 | 0.0007 | -0.0037 | 0.0007 | -0.0113 | 0.0016 | 1.0000 | -0.0105 |
| education_required_ord | -0.0075 | -0.0102 | -0.0089 | -0.0052 | 0.0003 | -0.0043 | 0.0028 | 0.0013 | -0.0030 | -0.0105 | 1.0000 |

### Biến tương quan dương cao với salary_usd
| feature | salary_usd |
| --- | --- |
| salary_local | 0.8772 |
| experience_level_ord | 0.7717 |
| years_experience | 0.7530 |
| remote_ratio | 0.0088 |
| days_to_deadline | 0.0004 |

### Biến tương quan âm/yếu với salary_usd
| feature | salary_usd |
| --- | --- |
| home_country_match | -0.0012 |
| benefits_score | -0.0017 |
| skills_count | -0.0052 |
| education_required_ord | -0.0075 |
| job_description_length | -0.0109 |

## 9) Nhận xét tổng hợp
- Dữ liệu sau tiền xử lý không còn giá trị thiếu.
- Có sự khác biệt rõ rệt về mức lương theo cấp độ kinh nghiệm.
- Biến years_experience và experience_level_ord có tương quan dương mạnh với salary_usd.
- Các biến remote_ratio, skills_count có tương quan yếu với lương.
- Cần kết hợp EDA với kết quả Regression/Classification để rút ra kết luận chặt chẽ.
