<?php
session_start();
include 'config.php';
include 'includes/language-loader.php';

// 未登录重定向
if (!isset($_SESSION['loggedin']) || $_SESSION['loggedin'] !== true) {
    header("Location: login.php");
    exit();
}

// 初始化变量
$success = '';
$error = '';

// 包含通知函数
include 'includes/notification-functions.php';

// 检查会话中是否有成功消息
if (isset($_SESSION['settings_success'])) {
    $success = $_SESSION['settings_success'];
    unset($_SESSION['settings_success']); // 清除会话中的消息
}

// 获取当前用户信息
$current_user_id = $_SESSION['id'];
$current_user_sql = "SELECT username, fullname, profile_picture, role FROM users WHERE employee_id = ?";
$stmt = $conn->prepare($current_user_sql);
$stmt->bind_param("i", $current_user_id);
$stmt->execute();
$current_user_result = $stmt->get_result();
$current_user = $current_user_result->fetch_assoc();

// 检查是否存在用户设置
$settings_sql = "SELECT * FROM users_settings WHERE employee_id = ?";
$stmt = $conn->prepare($settings_sql);
$stmt->bind_param("i", $current_user_id);
$stmt->execute();
$settings_result = $stmt->get_result();

// 默认设置
$settings = [
    'language' => 'english',
    'theme' => 'raspberry_pi'
];

// 如果有设置，则覆盖默认值
if ($settings_result->num_rows > 0) {
    $user_settings = $settings_result->fetch_assoc();
    $settings['language'] = $user_settings['language'];
    $settings['theme'] = $user_settings['theme'];
}

// 获取工作时间设置
$work_time_sql = "SELECT * FROM work_time_settings ORDER BY created_at DESC LIMIT 1";
$work_time_result = $conn->query($work_time_sql);
$work_time_settings = [
    'start_time' => '08:30:00',
    'end_time' => '18:00:00'
];

if ($work_time_result && $work_time_result->num_rows > 0) {
    $work_time = $work_time_result->fetch_assoc();
    $work_time_settings['start_time'] = $work_time['start_time'];
    $work_time_settings['end_time'] = $work_time['end_time'];
}

// 处理设置更新
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['save_settings'])) {
    // 获取表单数据
    $language = $_POST['language'];
    $theme = $_POST['theme'];

    // 获取工作时间设置
    $start_time = isset($_POST['start_time']) ? $_POST['start_time'] : '09:00:00';
    $end_time = isset($_POST['end_time']) ? $_POST['end_time'] : '17:00:00';

    // 开始事务
    $conn->begin_transaction();

    try {
        // 检查是否已有设置
        if ($settings_result->num_rows > 0) {
            // 更新现有设置
            $update_sql = "UPDATE users_settings SET
                          language = ?,
                          theme = ?
                          WHERE employee_id = ?";
            $update_stmt = $conn->prepare($update_sql);
            $update_stmt->bind_param("ssi", $language, $theme, $current_user_id);
            $update_stmt->execute();
        } else {
            // 插入新设置
            $insert_sql = "INSERT INTO users_settings (employee_id, language, theme)
                          VALUES (?, ?, ?)";
            $insert_stmt = $conn->prepare($insert_sql);
            $insert_stmt->bind_param("iss", $current_user_id, $language, $theme);
            $insert_stmt->execute();
        }

        // 更新工作时间设置
        if ($current_user['role'] === 'admin' && isset($_POST['update_work_time'])) {
            // 检查是否有现有记录
            $check_work_time_sql = "SELECT COUNT(*) as count FROM work_time_settings";
            $check_result = $conn->query($check_work_time_sql);
            $row = $check_result->fetch_assoc();

            if ($row['count'] > 0) {
                // 更新所有现有记录（因为我们只保留一条记录）
                $update_work_time_sql = "UPDATE work_time_settings SET start_time = ?, end_time = ?";
                $update_work_time_stmt = $conn->prepare($update_work_time_sql);
                $update_work_time_stmt->bind_param("ss", $start_time, $end_time);
                $update_work_time_stmt->execute();

                // 设置工作时间更新成功通知
                set_success_notification(__('work_time_updated'));
            } else {
                // 如果没有记录，则插入新记录
                $insert_work_time_sql = "INSERT INTO work_time_settings (start_time, end_time) VALUES (?, ?)";
                $insert_work_time_stmt = $conn->prepare($insert_work_time_sql);
                $insert_work_time_stmt->bind_param("ss", $start_time, $end_time);
                $insert_work_time_stmt->execute();

                // 设置工作时间创建成功通知
                set_success_notification(__('work_time_created'));
            }
        }

        // 提交事务
        $conn->commit();

        // 更新会话中的设置
        $_SESSION['settings'] = [
            'language' => $language,
            'theme' => $theme
        ];

        // 设置一个成功通知到会话中
        set_success_notification(__('settings_updated'));

        // 重定向到同一页面以刷新
        header("Location: settings.php");
        exit();

    } catch (Exception $e) {
        // 回滚事务
        $conn->rollback();
        $error = $e->getMessage();
    }
}

// 数据库表创建功能已移除
?>
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="icon.png" type="image/x-icon">
    <link rel="stylesheet" href="css/dashboard-style.css">
    <link rel="stylesheet" href="css/slidebar.css">
    <link rel="stylesheet" href="css/settings-style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <?php
    include 'includes/theme-loader.php';
    include 'includes/date-formatter.php';
    include 'includes/notifications-loader.php';
    ?>
    <title><?php echo __('settings'); ?> - FRAS System</title>
</head>
<body>
    <div class="dashboard-container">
        <?php include 'includes/sidebar.php'; ?>

        <!-- 主内容区 -->
        <div class="main-content">
            <!-- 顶部标题区 -->
            <div class="content-header">
                <h2 class="content-title"><?php echo __('settings'); ?></h2>

                <div class="user-profile">
                    <img src="data:image/jpeg;base64,<?php echo base64_encode($current_user['profile_picture']); ?>" alt="Profile Picture">
                </div>
            </div>

            <!-- 设置区域 -->
            <div class="settings-container">
                <?php if (!empty($error)): ?>
                    <script>
                        document.addEventListener('DOMContentLoaded', function() {
                            showErrorNotification('<?php echo addslashes($error); ?>');
                        });
                    </script>
                <?php endif; ?>

                <?php if (!empty($success)): ?>
                    <script>
                        document.addEventListener('DOMContentLoaded', function() {
                            showSuccessNotification('<?php echo addslashes($success); ?>');
                        });
                    </script>
                <?php endif; ?>

                <form method="POST">
                    <!-- 语言设置 -->
                    <div class="settings-section">
                        <h3><i class="fas fa-language"></i> <?php echo __('language_settings'); ?></h3>

                        <div class="form-group">
                            <label for="language"><?php echo __('interface_language'); ?>:</label>
                            <select id="language" name="language">
                                <option value="english" <?php echo $settings['language'] == 'english' ? 'selected' : ''; ?>>English</option>
                                <option value="chinese" <?php echo $settings['language'] == 'chinese' ? 'selected' : ''; ?>>Chinese (中文)</option>
                            </select>
                        </div>
                    </div>

                    <!-- 主题设置 -->
                    <div class="settings-section">
                        <h3><i class="fas fa-palette"></i> <?php echo __('theme_settings'); ?></h3>

                        <div class="form-group">
                            <label><?php echo __('select_theme'); ?>:</label>
                            <div class="theme-options">
                                <div class="theme-option raspberry-pi <?php echo $settings['theme'] == 'raspberry_pi' ? 'active' : ''; ?>" data-theme="raspberry_pi" onclick="selectTheme('raspberry_pi', this)">
                                    <div class="theme-preview">
                                        <div class="theme-sidebar"></div>
                                        <div class="theme-content"></div>
                                    </div>
                                    <div class="theme-name"><?php echo __('theme_raspberry_pi'); ?></div>
                                </div>

                                <div class="theme-option dark-theme <?php echo $settings['theme'] == 'dark' ? 'active' : ''; ?>" data-theme="dark" onclick="selectTheme('dark', this)">
                                    <div class="theme-preview">
                                        <div class="theme-sidebar"></div>
                                        <div class="theme-content"></div>
                                    </div>
                                    <div class="theme-name"><?php echo __('theme_dark'); ?></div>
                                </div>

                                <div class="theme-option fancy-theme <?php echo $settings['theme'] == 'fancy' ? 'active' : ''; ?>" data-theme="fancy" onclick="selectTheme('fancy', this)">
                                    <div class="theme-preview">
                                        <div class="theme-sidebar"></div>
                                        <div class="theme-content"></div>
                                    </div>
                                    <div class="theme-name"><?php echo __('theme_fancy'); ?></div>
                                </div>

                                <div class="theme-option ocean-theme <?php echo $settings['theme'] == 'ocean' ? 'active' : ''; ?>" data-theme="ocean" onclick="selectTheme('ocean', this)">
                                    <div class="theme-preview">
                                        <div class="theme-sidebar"></div>
                                        <div class="theme-content"></div>
                                    </div>
                                    <div class="theme-name"><?php echo __('theme_ocean'); ?></div>
                                </div>
                            </div>
                            <input type="hidden" id="theme" name="theme" value="<?php echo $settings['theme']; ?>">
                        </div>
                    </div>

                    <?php if ($current_user['role'] === 'admin'): ?>
                    <!-- 工作时间设置 -->
                    <div class="settings-section">
                        <h3><i class="fas fa-business-time"></i> <?php echo __('work_time_settings'); ?></h3>

                        <div class="form-group">
                            <label for="start_time"><?php echo __('work_start_time'); ?>:</label>
                            <input type="time" id="start_time" name="start_time" value="<?php echo substr($work_time_settings['start_time'], 0, 5); ?>" step="300" required>
                        </div>

                        <div class="form-group">
                            <label for="end_time"><?php echo __('work_end_time'); ?>:</label>
                            <input type="time" id="end_time" name="end_time" value="<?php echo substr($work_time_settings['end_time'], 0, 5); ?>" step="300" required>
                        </div>

                        <div class="form-group">
                            <div class="work-hours-preview">
                                <span class="preview-label"><?php echo __('work_hours_preview'); ?>:</span>
                                <span id="workHoursDisplay" class="preview-value"></span>
                            </div>
                        </div>

                        <div class="form-group">
                            <div class="checkbox-container">
                                <input type="checkbox" id="update_work_time" name="update_work_time" value="1">
                                <label for="update_work_time"><?php echo __('update_work_time'); ?></label>
                            </div>
                            <p class="help-text"><?php echo __('work_time_help'); ?></p>
                        </div>
                    </div>

                    <script>
                        document.addEventListener('DOMContentLoaded', function() {
                            const startTimeInput = document.getElementById('start_time');
                            const endTimeInput = document.getElementById('end_time');
                            const workHoursDisplay = document.getElementById('workHoursDisplay');

                            // 初始计算工作时长
                            calculateWorkHours();

                            // 添加事件监听器
                            startTimeInput.addEventListener('change', calculateWorkHours);
                            endTimeInput.addEventListener('change', calculateWorkHours);

                            function calculateWorkHours() {
                                const startTime = startTimeInput.value;
                                const endTime = endTimeInput.value;

                                if (startTime && endTime) {
                                    // 转换为分钟
                                    const [startHours, startMinutes] = startTime.split(':').map(Number);
                                    const [endHours, endMinutes] = endTime.split(':').map(Number);

                                    let startTotalMinutes = startHours * 60 + startMinutes;
                                    let endTotalMinutes = endHours * 60 + endMinutes;

                                    // 处理跨天情况
                                    if (endTotalMinutes < startTotalMinutes) {
                                        endTotalMinutes += 24 * 60; // 加一天的分钟数
                                    }

                                    // 计算差值
                                    const diffMinutes = endTotalMinutes - startTotalMinutes;
                                    const hours = Math.floor(diffMinutes / 60);
                                    const minutes = diffMinutes % 60;

                                    // 格式化显示
                                    workHoursDisplay.textContent = `${hours} <?php echo __('hours'); ?> ${minutes} <?php echo __('minutes'); ?>`;

                                    // 根据工作时长设置样式
                                    if (diffMinutes < 480) { // 少于8小时
                                        workHoursDisplay.className = 'preview-value short-hours';
                                    } else if (diffMinutes > 600) { // 多于10小时
                                        workHoursDisplay.className = 'preview-value long-hours';
                                    } else { // 8-10小时，正常范围
                                        workHoursDisplay.className = 'preview-value normal-hours';
                                    }
                                }
                            }
                        });
                    </script>
                    <?php endif; ?>

                    <div class="button-group">
                        <a href="dashboard.php" class="cancel-button">
                            <i class="fas fa-times"></i> <?php echo __('cancel'); ?>
                        </a>
                        <button type="submit" name="save_settings" class="save-button">
                            <i class="fas fa-save"></i> <?php echo __('save_settings'); ?>
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <!-- 引入主题CSS和JS -->
    <link rel="stylesheet" href="css/themes.css">
    <script src="js/theme-switcher.js"></script>
    <script>
        // 选择主题
        function selectTheme(theme, element) {
            // 更新隐藏输入
            document.getElementById('theme').value = theme;

            // 更新活动类
            const themeOptions = document.querySelectorAll('.theme-option');
            themeOptions.forEach(option => {
                option.classList.remove('active');
            });

            element.classList.add('active');

            // 应用主题（立即预览）
            applyTheme(theme);
        }
    </script>
</body>
</html>
