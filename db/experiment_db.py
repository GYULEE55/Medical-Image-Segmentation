"""
실험 결과 저장용 SQLite DB

MLflow 쓰기엔 오버킬이라 가볍게 SQLite로 직접 구현.
외부 의존성 없이 stdlib만 사용.

Tables:
    experiments  - 실험 기본 정보 (모델, 하이퍼파라미터 등)
    metrics      - 성능 지표 (mAP, loss 등)
    predictions  - 개별 이미지 추론 결과
"""

import sqlite3
import json
from pathlib import Path


class ExperimentDB:
    """SQLite 기반 실험 결과 관리"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent / "experiments.db")

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        print(f"[DB] 연결: {db_path}")

    def _create_tables(self):
        """테이블 초기화"""
        cur = self.conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                model       TEXT NOT NULL,
                dataset     TEXT NOT NULL,
                epochs      INTEGER,
                batch_size  INTEGER,
                imgsz       INTEGER,
                lr          REAL,
                optimizer   TEXT,
                notes       TEXT,
                config_json TEXT,
                created_at  TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                metric_name   TEXT NOT NULL,
                metric_value  REAL NOT NULL,
                epoch         INTEGER,  -- NULL이면 최종 결과
                created_at    TEXT DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                image_name    TEXT NOT NULL,
                confidence    REAL,
                class_name    TEXT,
                bbox_json     TEXT,
                created_at    TEXT DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)

        self.conn.commit()

    def add_experiment(self, name, model, dataset, epochs=None,
                       batch_size=None, imgsz=640, lr=None,
                       optimizer=None, notes=None, **extra_config):
        """실험 등록 후 experiment_id 반환"""
        config_json = json.dumps(extra_config, ensure_ascii=False) if extra_config else None

        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO experiments
                (name, model, dataset, epochs, batch_size, imgsz, lr, optimizer, notes, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, model, dataset, epochs, batch_size, imgsz, lr, optimizer, notes, config_json))
        self.conn.commit()

        exp_id = cur.lastrowid
        print(f"[DB] 실험 #{exp_id} '{name}' 등록")
        return exp_id

    def add_metrics(self, experiment_id, epoch=None, **metrics):
        """
        성능 지표 저장. kwargs로 지표명=값 전달.
        >>> db.add_metrics(exp_id, map50=0.85, map50_95=0.62)
        """
        cur = self.conn.cursor()
        for name, value in metrics.items():
            cur.execute(
                "INSERT INTO metrics (experiment_id, metric_name, metric_value, epoch) "
                "VALUES (?, ?, ?, ?)",
                (experiment_id, name, float(value), epoch),
            )
        self.conn.commit()
        print(f"[DB] 지표 {len(metrics)}개 저장 (실험 #{experiment_id})")

    def add_prediction(self, experiment_id, image_name,
                       confidence=None, class_name=None, bbox=None):
        """개별 이미지 추론 결과 저장"""
        bbox_json = json.dumps(bbox) if bbox else None
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO predictions "
            "(experiment_id, image_name, confidence, class_name, bbox_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (experiment_id, image_name, confidence, class_name, bbox_json),
        )
        self.conn.commit()

    def get_best_experiment(self, metric_name="map50"):
        """지표 기준 최고 성능 실험 조회"""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT e.*, m.metric_value, m.metric_name
            FROM experiments e
            JOIN metrics m ON e.id = m.experiment_id
            WHERE m.metric_name = ?
            ORDER BY m.metric_value DESC
            LIMIT 1
        """, (metric_name,))
        row = cur.fetchone()
        return dict(row) if row else None

    def compare_experiments(self):
        """전체 실험 비교. 각 실험의 최종 지표도 같이 반환."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM experiments ORDER BY created_at DESC")
        experiments = [dict(row) for row in cur.fetchall()]

        for exp in experiments:
            cur.execute(
                "SELECT metric_name, metric_value FROM metrics "
                "WHERE experiment_id = ? AND epoch IS NULL",
                (exp["id"],),
            )
            for row in cur.fetchall():
                exp[row["metric_name"]] = row["metric_value"]

        return experiments

    def get_experiment_history(self, experiment_id):
        """에포크별 지표 변화 (학습 곡선 분석용)"""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT epoch, metric_name, metric_value
            FROM metrics
            WHERE experiment_id = ? AND epoch IS NOT NULL
            ORDER BY epoch, metric_name
        """, (experiment_id,))
        return [dict(row) for row in cur.fetchall()]

    def run_query(self, sql, params=()):
        """자유 SQL 쿼리. 파라미터 바인딩으로 injection 방지."""
        cur = self.conn.cursor()
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def close(self):
        self.conn.close()
        print("[DB] 연결 종료")

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    import csv

    db = ExperimentDB()

    # 실험 등록
    exp_id = db.add_experiment(
        name="kvasir-seg-v1",
        model="yolov8n-seg",
        dataset="kvasir-seg",
        epochs=50,
        batch_size=16,
        imgsz=640,
        lr=0.01,
        optimizer="SGD",
        notes="Colab T4 GPU. patience=10.",
        patience=10,
    )

    # results.csv → DB 저장 (에포크별 + 최종)
    results_csv = Path(__file__).parent.parent / "results.csv"
    if not results_csv.exists():
        print(f"[오류] results.csv 없음: {results_csv}")
        db.close()
        exit(1)

    last_row = None
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            if not row.get("epoch"):
                continue

            epoch = int(row["epoch"])
            db.add_metrics(
                exp_id,
                epoch=epoch,
                map50=row["metrics/mAP50(B)"],
                map50_95=row["metrics/mAP50-95(B)"],
                precision=row["metrics/precision(B)"],
                recall=row["metrics/recall(B)"],
                map50_mask=row["metrics/mAP50(M)"],
                map50_95_mask=row["metrics/mAP50-95(M)"],
            )
            last_row = row

    # 최종 결과 (epoch=None → compare_experiments에서 사용)
    if last_row:
        db.add_metrics(
            exp_id,
            map50=last_row["metrics/mAP50(B)"],
            map50_95=last_row["metrics/mAP50-95(B)"],
            precision=last_row["metrics/precision(B)"],
            recall=last_row["metrics/recall(B)"],
            map50_mask=last_row["metrics/mAP50(M)"],
            map50_95_mask=last_row["metrics/mAP50-95(M)"],
        )

    # 결과 확인
    print("\n전체 실험:")
    for exp in db.compare_experiments():
        print(f"  [{exp['id']}] {exp['name']} | {exp['model']} | "
              f"mAP50={exp.get('map50', '-')} | mAP50-95={exp.get('map50_95', '-')}")

    history = db.get_experiment_history(exp_id)
    print(f"\n에포크별 기록: {len(history)}개")

    db.close()
