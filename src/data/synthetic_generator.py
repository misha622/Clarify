"""
Генератор синтетических данных для Clarify.

Генерирует:
- Benign-трафик: легитимный DNS, нерегулярные всплески, нормальная активность
- Attack-сценарии: brute-force (RDP/SSH), beaconing (C2), DGA (DNS)
- Два режима: 70/30 для обучения, ~99/1 для калибровки порогов
- Достаточная история для оконных фич (≥15 интервалов на окно)
- Нарезка на скользящие окна внутри хоста для обучения

Использование:
    generator = SyntheticGenerator(seed=42)
    data = generator.generate(mode="train")
    X, y = generator.generate_for_beaconing_training(mode="train")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class SyntheticEvent:
    """Одно событие в синтетическом датасете."""
    timestamp: float
    source_ip: str
    event_type: str
    label: int  # 0 = benign, 1 = attack
    metadata: dict = field(default_factory=dict)


@dataclass
class SyntheticDataset:
    """Сгенерированный датасет."""
    events: list[SyntheticEvent]
    mode: str
    total_events: int
    benign_count: int
    attack_count: int

    @property
    def attack_ratio(self) -> float:
        return self.attack_count / self.total_events if self.total_events > 0 else 0.0


class SyntheticGenerator:
    """
    Генератор синтетических данных для обучения и калибровки Clarify.

    Генерирует реалистичный сетевой трафик с нормальной активностью
    и тремя типами атак.

    Для обучения нарезает трафик каждого хоста на скользящие окна,
    что даёт сотни сэмплов из ограниченного числа хостов.
    """

    LEGITIMATE_DOMAINS = [
        "google.com", "github.com", "amazonaws.com", "cloudflare.com",
        "microsoft.com", "apple.com", "facebook.com", "twitter.com",
        "stackoverflow.com", "pypi.org", "docker.com", "kubernetes.io",
        "redhat.com", "ubuntu.com", "debian.org", "nginx.org",
        "python.org", "npmjs.com", "wikipedia.org", "archive.org",
        "gitlab.com", "bitbucket.org", "mozilla.org", "apache.org",
        "grafana.com", "prometheus.io", "elastic.co", "hashicorp.com",
        "digitalocean.com", "linode.com",
    ]

    def __init__(
            self,
            seed: int = 42,
            time_window_hours: int = 6,
    ):
        """
        Args:
            seed: сид для воспроизводимости
            time_window_hours: размер временного окна в часах
        """
        self.rng = np.random.RandomState(seed)
        self.time_window = time_window_hours * 3600
        self.base_time = 1_700_000_000.0

    def _random_ip(self, prefix: str = "10.0") -> str:
        return f"{prefix}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}"

    def _random_public_ip(self) -> str:
        return f"{self.rng.randint(1, 255)}.{self.rng.randint(0, 255)}.{self.rng.randint(0, 255)}.{self.rng.randint(1, 255)}"

    def _generate_dga_domain(self) -> str:
        length = self.rng.randint(12, 25)
        chars = list("abcdefghijklmnopqrstuvwxyz0123456789")
        name = "".join(self.rng.choice(chars, size=length))
        tlds = [".com", ".net", ".xyz", ".top", ".pw", ".info", ".biz"]
        return name + self.rng.choice(tlds)

    # ------------------------------------------------------------------
    # Benign-генераторы
    # ------------------------------------------------------------------

    def _generate_legitimate_dns(
            self, source_ip: str, count: int
    ) -> list[SyntheticEvent]:
        """Генерирует легитимный DNS-трафик от одного источника."""
        events = []
        current_time = self.base_time + self.rng.uniform(0, self.time_window * 0.1)

        for _ in range(count):
            domain = self.rng.choice(self.LEGITIMATE_DOMAINS)
            events.append(SyntheticEvent(
                timestamp=current_time,
                source_ip=source_ip,
                event_type="dns_query",
                label=0,
                metadata={"domain": domain, "nxdomain": False},
            ))
            current_time += self.rng.exponential(45.0)
            if current_time > self.base_time + self.time_window:
                break

        return events

    def _generate_irregular_surge(
            self, source_ip: str, count: int
    ) -> list[SyntheticEvent]:
        """Генерирует нерегулярный всплеск трафика (легитимный)."""
        events = []
        surge_start = self.base_time + self.rng.uniform(
            self.time_window * 0.2, self.time_window * 0.8
        )
        current_time = surge_start

        for i in range(count):
            events.append(SyntheticEvent(
                timestamp=current_time,
                source_ip=source_ip,
                event_type="http_request",
                label=0,
                metadata={"url": f"/api/endpoint_{self.rng.randint(1, 100)}"},
            ))
            current_time += self.rng.exponential(25.0)

        return events

    # ------------------------------------------------------------------
    # Attack-генераторы
    # ------------------------------------------------------------------

    def _generate_brute_force(
            self, source_ip: str, target_ip: str, attempts: int
    ) -> list[SyntheticEvent]:
        """Генерирует brute-force атаку."""
        events = []
        attack_start = self.base_time + self.rng.uniform(
            self.time_window * 0.2, self.time_window * 0.6
        )
        current_time = attack_start

        for i in range(max(attempts, 30)):
            events.append(SyntheticEvent(
                timestamp=current_time,
                source_ip=source_ip,
                event_type="auth_failure",
                label=1,
                metadata={
                    "target_ip": target_ip,
                    "protocol": self.rng.choice(["RDP", "SSH", "FTP"]),
                    "attempt": i + 1,
                    "username": f"admin_{self.rng.randint(1, 100)}",
                },
            ))
            current_time += 1.5 + abs(self.rng.normal(0, 0.2))

        return events

    def _generate_beaconing(
            self, source_ip: str, beacons: int
    ) -> list[SyntheticEvent]:
        """Генерирует C2 Beaconing."""
        events = []
        beacon_start = self.base_time + self.rng.uniform(0, self.time_window * 0.3)
        current_time = beacon_start

        beacon_interval = self.rng.choice([60.0, 90.0, 120.0, 137.0, 180.0])

        for _ in range(max(beacons, 30)):
            events.append(SyntheticEvent(
                timestamp=current_time,
                source_ip=source_ip,
                event_type="dns_query",
                label=1,
                metadata={
                    "domain": f"beacon-{self.rng.randint(1000, 9999)}.evil-c2.com",
                    "nxdomain": True,
                    "beacon_type": "C2",
                },
            ))
            current_time += beacon_interval + self.rng.normal(0, 0.05 * beacon_interval)

        return events

    def _generate_dga(
            self, source_ip: str, domains: int
    ) -> list[SyntheticEvent]:
        """Генерирует DGA-атаку."""
        events = []
        dga_start = self.base_time + self.rng.uniform(0, self.time_window * 0.3)
        current_time = dga_start

        for _ in range(max(domains, 60)):
            domain = self._generate_dga_domain()
            queries = self.rng.randint(1, 3)
            for _ in range(queries):
                events.append(SyntheticEvent(
                    timestamp=current_time,
                    source_ip=source_ip,
                    event_type="dns_query",
                    label=1,
                    metadata={
                        "domain": domain,
                        "nxdomain": True,
                        "entropy": self.rng.uniform(0.75, 0.98),
                    },
                ))
                current_time += self.rng.uniform(0.05, 1.5)

        return events

    # ------------------------------------------------------------------
    # Основной генератор событий
    # ------------------------------------------------------------------

    def generate(
            self,
            mode: Literal["train", "calibrate"] = "train",
            num_hosts: int = 30,
    ) -> SyntheticDataset:
        """
        Генерирует полный синтетический датасет событий.

        Args:
            mode: "train" (70/30) или "calibrate" (~99/1 по хостам)
            num_hosts: количество уникальных хостов

        Returns:
            SyntheticDataset с событиями и метаданными
        """
        events: list[SyntheticEvent] = []

        if mode == "train":
            benign_hosts = int(num_hosts * 0.7)
            attack_hosts = num_hosts - benign_hosts
        else:
            benign_hosts = max(int(num_hosts * 0.97), num_hosts - 3)
            attack_hosts = num_hosts - benign_hosts

        # Benign-трафик
        for i in range(benign_hosts):
            source_ip = self._random_ip(f"10.{10 + (i // 250)}")
            dns_count = self.rng.randint(30, 60)
            events.extend(self._generate_legitimate_dns(source_ip, dns_count))

            if self.rng.random() < 0.3:
                surge_count = self.rng.randint(20, 35)
                events.extend(self._generate_irregular_surge(source_ip, surge_count))

        # Атаки
        for i in range(attack_hosts):
            attack_ip = self._random_public_ip()
            attack_type = self.rng.choice(["brute_force", "beaconing", "dga"])

            if attack_type == "brute_force":
                target = self._random_ip("192.168")
                events.extend(self._generate_brute_force(attack_ip, target, attempts=35))
            elif attack_type == "beaconing":
                events.extend(self._generate_beaconing(attack_ip, beacons=35))
            else:
                events.extend(self._generate_dga(attack_ip, domains=65))

        events.sort(key=lambda e: e.timestamp)

        benign_count = sum(1 for e in events if e.label == 0)
        attack_count = sum(1 for e in events if e.label == 1)

        return SyntheticDataset(
            events=events,
            mode=mode,
            total_events=len(events),
            benign_count=benign_count,
            attack_count=attack_count,
        )

    # ------------------------------------------------------------------
    # Генератор для обучения: нарезка на скользящие окна
    # ------------------------------------------------------------------

    def _slice_into_windows(
            self,
            timestamps: list[float],
            window_size_seconds: float = 1800.0,
            stride_seconds: float = 600.0,
            min_events_per_window: int = 10,
    ) -> list[list[float]]:
        """
        Нарезает временной ряд на перекрывающиеся окна.

        Args:
            timestamps: отсортированные временные метки
            window_size_seconds: размер окна
            stride_seconds: шаг смещения окна
            min_events_per_window: минимальное число событий в окне

        Returns:
            список окон, каждое окно — список временных меток внутри него
        """
        if len(timestamps) < min_events_per_window:
            return []

        t_min = timestamps[0]
        t_max = timestamps[-1]

        windows = []
        window_start = t_min

        while window_start + window_size_seconds <= t_max:
            window_end = window_start + window_size_seconds
            window_events = [
                t for t in timestamps
                if window_start <= t <= window_end
            ]

            if len(window_events) >= min_events_per_window:
                windows.append(window_events)

            window_start += stride_seconds

        return windows

    def generate_for_beaconing_training(
            self,
            mode: Literal["train", "calibrate"] = "train",
            window_size_seconds: float = 1800.0,
            stride_seconds: float = 600.0,
            min_events_per_window: int = 10,
            num_hosts: int = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Генерирует данные для обучения Beaconing-детектора.

        Нарезает трафик каждого хоста на скользящие окна,
        что даёт сотни сэмплов из ограниченного числа хостов.

        Args:
            mode: "train" (70/30) или "calibrate" (~99/1)
            window_size_seconds: размер окна в секундах
            stride_seconds: шаг окна в секундах
            min_events_per_window: минимум событий в окне
            num_hosts: количество хостов (None = авто: 100 для train, 300 для calibrate)

        Returns:
            (X, y) — признаки и метки
        """
        from src.features.window_stats import calculate_window_stats

        if num_hosts is None:
            num_hosts = 100 if mode == "train" else 300

        dataset = self.generate(mode=mode, num_hosts=num_hosts)

        # Группируем события по source_ip
        sources: dict[str, dict] = {}

        for event in dataset.events:
            if event.source_ip not in sources:
                sources[event.source_ip] = {
                    "timestamps": [],
                    "label": event.label,
                }
            sources[event.source_ip]["timestamps"].append(event.timestamp)

        X_list, y_list = [], []

        for ip, data in sources.items():
            timestamps = data["timestamps"]
            label = data["label"]

            windows = self._slice_into_windows(
                timestamps,
                window_size_seconds=window_size_seconds,
                stride_seconds=stride_seconds,
                min_events_per_window=min_events_per_window,
            )

            for window_ts in windows:
                stats = calculate_window_stats(window_ts)
                X_list.append(stats.to_feature_vector())
                y_list.append(label)

        X = np.array(X_list)
        y = np.array(y_list)

        # Перемешиваем
        if len(X) > 0:
            idx = self.rng.permutation(len(X))
            return X[idx], y[idx]

        return X, y


# ------------------------------------------------------------------
# Быстрый тест
# ------------------------------------------------------------------
if __name__ == "__main__":
    generator = SyntheticGenerator(seed=42)

    print("=" * 70)
    print("Тест генератора синтетики с нарезкой окон")
    print()

    print("1. Train-датасет (70/30):")
    dataset = generator.generate(mode="train", num_hosts=50)
    print(f"   Событий: {dataset.total_events}")
    print(f"   Benign: {dataset.benign_count}")
    print(f"   Attack: {dataset.attack_count}")
    print(f"   Соотношение атак: {dataset.attack_ratio:.1%}")

    X_train, y_train = generator.generate_for_beaconing_training(
        mode="train", window_size_seconds=900, stride_seconds=300,
        min_events_per_window=8, num_hosts=100,
    )
    print(f"   Окон (сэмплов): {len(X_train)}")
    print(f"   Распределение: benign={sum(y_train == 0)}, attack={sum(y_train == 1)}")

    print()

    print("2. Calibrate-датасет (~99/1):")
    dataset_cal = generator.generate(mode="calibrate", num_hosts=100)
    print(f"   Событий: {dataset_cal.total_events}")
    print(f"   Benign: {dataset_cal.benign_count}")
    print(f"   Attack: {dataset_cal.attack_count}")
    print(f"   Соотношение атак: {dataset_cal.attack_ratio:.1%}")

    X_cal, y_cal = generator.generate_for_beaconing_training(
        mode="calibrate", window_size_seconds=900, stride_seconds=300,
        min_events_per_window=8, num_hosts=300,
    )
    print(f"   Окон (сэмплов): {len(X_cal)}")
    print(f"   Распределение: benign={sum(y_cal == 0)}, attack={sum(y_cal == 1)}")

    print()

    if len(X_train) > 0:
        print("3. Статистики (первые 3 benign-окна):")
        benign_idx = [j for j, label in enumerate(y_train) if label == 0][:3]
        for i, idx in enumerate(benign_idx):
            print(
                f"   Окно {i + 1}: CV={X_train[idx][2]:.3f}, entropy={X_train[idx][5]:.3f}, events={int(X_train[idx][6])}")

        print("4. Статистики (первые 3 attack-окна):")
        attack_idx = [j for j, label in enumerate(y_train) if label == 1][:3]
        for i, idx in enumerate(attack_idx):
            print(
                f"   Окно {i + 1}: CV={X_train[idx][2]:.3f}, entropy={X_train[idx][5]:.3f}, events={int(X_train[idx][6])}")

    print("=" * 70)