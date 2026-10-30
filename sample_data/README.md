# Sample Bluebook Datasets

This directory contains test examination sheets representing realistic scenarios encountered during college IA evaluations:

1. `clean_bluebook_template.png`: Blank reference copy of the college continuous internal assessment second sheet evaluation grid.
2. `student_sample_1.png`: Cleanly evaluated student assessment with standard marks awarded.
3. `student_sample_2_skewed.jpg`: Evaluated sheet with a +3.8 degree rotation and scaled dimensions to verify automatic deskewing.
4. `student_sample_3_perspective.png`: Evaluated sheet exhibiting mobile phone perspective angle distortion.
5. `student_sample_4_multipage.pdf`: Two-page scanned PDF consisting of Page 1 (Cover Page) and Page 2 (Marks Table Sheet).
6. `student_sample_5_review_needed.png`: Contains edge cases such as an invalid mark exceeding question max marks and an unattempted blank question.

To regenerate these files at any time, run:
```bash
python sample_data/generate_samples.py
```
