from datetime import datetime

from extensions import db


class Dataset(db.Model):
    __tablename__ = "datasets"

    id = db.Column(db.Integer, primary_key=True)

    # Generic source information
    source_type = db.Column(
        db.String(50),
        nullable=False,
        default="file"
    )

    source_name = db.Column(
        db.String(255),
        nullable=False
    )

    # File-specific information
    original_filename = db.Column(
        db.String(255),
        nullable=True
    )

    stored_filename = db.Column(
        db.String(255),
        nullable=True,
        unique=True
    )

    file_path = db.Column(
        db.String(500),
        nullable=True
    )

    file_type = db.Column(
        db.String(20),
        nullable=True
    )

    # Generic dataset metadata
    row_count = db.Column(
        db.Integer,
        nullable=True
    )

    column_count = db.Column(
        db.Integer,
        nullable=True
    )

    uploaded_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )