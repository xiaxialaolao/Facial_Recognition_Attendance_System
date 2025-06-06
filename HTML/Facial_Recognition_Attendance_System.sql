-- 数据库创建语句已移除
-- 请确保已经创建了数据库并选择了正确的数据库

CREATE TABLE users (
    employee_id INT(8) PRIMARY KEY NOT NULL,
    username VARCHAR(30) NOT NULL,
    password VARCHAR(255) NOT NULL,
    fullname VARCHAR(255) NOT NULL,
    role ENUM('user', 'admin') DEFAULT 'user' NOT NULL,
    profile_picture Longblob NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE attendance (
    attendance_id INT PRIMARY KEY AUTO_INCREMENT,
    employee_id INT(8) NOT NULL,
    session_type ENUM('in', 'out') NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (employee_id) REFERENCES users(employee_id)
);

CREATE TABLE users_settings (
    employee_id INT(8) NOT NULL,
    language VARCHAR(50) NOT NULL DEFAULT 'english',
    theme VARCHAR(50) NOT NULL DEFAULT 'raspberry_pi',
    FOREIGN KEY (employee_id) REFERENCES users(employee_id)
);

CREATE TABLE work_time_settings (
        start_time TIME NOT NULL DEFAULT '08:30:00',
        end_time TIME NOT NULL DEFAULT '18:00:00',
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE system_logs (
        log_id INT PRIMARY KEY AUTO_INCREMENT,
        log_level ENUM('info', 'warning', 'error', 'debug') NOT NULL DEFAULT 'info',
        message TEXT NOT NULL,
        source VARCHAR(100),
        user_id INT(8),
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(employee_id) ON DELETE SET NULL
);

