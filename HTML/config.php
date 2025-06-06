<?php
// 数据库连接配置
$servername = "127.0.0.1";
$username = "xiaxialaolao";
$password = "xiaxialaolao";
$dbname = "Facial_Recognition_Attendance_System";

// 使用 mysqli 创建连接（在 dashboard.php 中使用）
$conn = new mysqli($servername, $username, $password, $dbname);

// 检查连接是否成功
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// 检查并创建必要的数据库表
function checkAndCreateTables($conn) {
    // 定义所有需要检查的表
    $tables = [
        'users' => [
            'check_query' => "SHOW COLUMNS FROM users",
            'expected_columns' => ['employee_id', 'username', 'password', 'fullname', 'role', 'profile_picture', 'created_at', 'updated_at'],
            'create_query' => "CREATE TABLE users (
                employee_id INT(8) PRIMARY KEY NOT NULL,
                username VARCHAR(30) NOT NULL,
                password VARCHAR(255) NOT NULL,
                fullname VARCHAR(255) NOT NULL,
                role ENUM('user', 'admin') DEFAULT 'user' NOT NULL,
                profile_picture Longblob NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )"
        ],
        'attendance' => [
            'check_query' => "SHOW COLUMNS FROM attendance",
            'expected_columns' => ['attendance_id', 'employee_id', 'session_type', 'created_at'],
            'create_query' => "CREATE TABLE attendance (
                attendance_id INT PRIMARY KEY AUTO_INCREMENT,
                employee_id INT(8) NOT NULL,
                session_type ENUM('in', 'out') NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES users(employee_id)
            )"
        ],
        'users_settings' => [
            'check_query' => "SHOW COLUMNS FROM users_settings",
            'expected_columns' => ['employee_id', 'language', 'theme'],
            'create_query' => "CREATE TABLE users_settings (
                employee_id INT(8) NOT NULL,
                language VARCHAR(50) NOT NULL DEFAULT 'english',
                theme VARCHAR(50) NOT NULL DEFAULT 'raspberry_pi',
                FOREIGN KEY (employee_id) REFERENCES users(employee_id)
            )"
        ],
        'work_time_settings' => [
            'check_query' => "SHOW COLUMNS FROM work_time_settings",
            'expected_columns' => ['start_time', 'end_time', 'created_at', 'updated_at'],
            'create_query' => "CREATE TABLE work_time_settings (
                start_time TIME NOT NULL DEFAULT '08:30:00',
                end_time TIME NOT NULL DEFAULT '18:00:00',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )"
        ],
        'system_logs' => [
            'check_query' => "SHOW COLUMNS FROM system_logs",
            'expected_columns' => ['log_id', 'log_level', 'message', 'source', 'user_id', 'created_at'],
            'create_query' => "CREATE TABLE system_logs (
                log_id INT PRIMARY KEY AUTO_INCREMENT,
                log_level ENUM('info', 'warning', 'error', 'debug') NOT NULL DEFAULT 'info',
                message TEXT NOT NULL,
                source VARCHAR(100),
                user_id INT(8),
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(employee_id) ON DELETE SET NULL
            )"
        ]
    ];

    // 检查每个表
    foreach ($tables as $table_name => $table_info) {
        // 检查表是否存在
        $table_exists = $conn->query("SHOW TABLES LIKE '$table_name'")->num_rows > 0;

        if (!$table_exists) {
            // 表不存在，创建它
            $conn->query($table_info['create_query']);
            continue;
        }

        // 表存在，检查结构是否完整
        $result = $conn->query($table_info['check_query']);
        $columns = [];

        while ($row = $result->fetch_assoc()) {
            $columns[] = $row['Field'];
        }

        // 检查是否缺少任何预期的列
        $missing_columns = array_diff($table_info['expected_columns'], $columns);

        if (!empty($missing_columns)) {
            // 结构不完整，删除并重新创建
            $conn->query("DROP TABLE $table_name");
            $conn->query($table_info['create_query']);
        }
    }

    // 检查 work_time_settings 表是否有数据，如果没有则插入默认值
    $work_time_count = $conn->query("SELECT COUNT(*) as count FROM work_time_settings")->fetch_assoc()['count'];
    if ($work_time_count == 0) {
        $conn->query("INSERT INTO work_time_settings (start_time, end_time) VALUES ('08:30:00', '18:00:00')");
    }
}

// 执行表检查和创建
checkAndCreateTables($conn);
?>
