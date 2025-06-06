<?php
session_start();
include 'config.php';
include 'includes/notification-functions.php';
include 'includes/language-loader.php';

// XSS过滤函数 - 检测并过滤潜在的XSS攻击
function filterXSS($input) {
    // 如果输入为空，直接返回
    if (empty($input)) {
        return $input;
    }

    // 检测常见的XSS攻击模式
    $patterns = [
        // 脚本标签
        '/<script[^>]*>.*?<\/script>/i',
        // 内联事件处理器
        '/on\w+\s*=\s*"[^"]*"/i',
        '/on\w+\s*=\s*\'[^\']*\'/i',
        // javascript: 协议
        '/javascript\s*:/i',
        // 其他危险标签
        '/<iframe[^>]*>.*?<\/iframe>/i',
        '/<object[^>]*>.*?<\/object>/i',
        '/<embed[^>]*>.*?<\/embed>/i',
        // 数据URI
        '/data\s*:[^\s,;]+/i'
    ];

    // 如果检测到XSS模式，返回false
    foreach ($patterns as $pattern) {
        if (preg_match($pattern, $input)) {
            return false;
        }
    }

    // 通过检测，返回过滤后的输入
    return $input;
}

// 高亮搜索词的函数
function highlightSearchTerm($text, $searchTerm, $searchField) {
    if (empty($searchTerm) || strlen($searchTerm) < 2) {
        return htmlspecialchars($text);
    }

    // 如果搜索字段是"all"或与当前字段匹配
    if ($searchField == 'all' ||
        ($searchField == 'employee_id' && is_numeric($text)) ||
        ($searchField == 'username' && !is_numeric($text)) ||
        ($searchField == 'fullname' && !is_numeric($text)) ||
        ($searchField == 'role' && ($text == 'admin' || $text == 'user' || $text == 'Admin' || $text == 'User'))) {

        $escapedText = htmlspecialchars($text);
        $escapedSearchTerm = htmlspecialchars($searchTerm);

        // 不区分大小写的替换
        $pattern = '/(' . preg_quote($escapedSearchTerm, '/') . ')/i';
        return preg_replace($pattern, '<span class="highlight">$1</span>', $escapedText);
    }

    return htmlspecialchars($text);
}

// 未登录重定向
if (!isset($_SESSION['loggedin']) || $_SESSION['loggedin'] !== true) {
    header("Location: login.php");
    exit();
}

// 初始化变量
$search_term = '';
$search_field = 'all';
$users = [];
$error = '';
$success = '';
$total_results = 0;

// 处理搜索请求
if ($_SERVER["REQUEST_METHOD"] == "GET" && isset($_GET['search'])) {
    $search_term = trim($_GET['search_term']);
    $search_field = isset($_GET['search_field']) ? $_GET['search_field'] : 'all';

    // XSS防护 - 检查搜索字段
    $filtered_field = filterXSS($search_field);
    if ($filtered_field === false) {
        $error = __('search_field_contains_html');
        set_error_notification($error);
        $search_field = 'all'; // 重置为安全值
    } else {
        $search_field = $filtered_field;
    }

    // XSS防护 - 检查搜索词
    $filtered_term = filterXSS($search_term);
    if ($filtered_term === false) {
        $error = __('search_term_contains_html');
        set_error_notification($error);
        $search_term = ''; // 清空搜索词
    } else {
        $search_term = $filtered_term;
    }

    // 验证搜索词长度
    if (strlen($search_term) < 2 && !empty($search_term)) {
        $error = __('search_term_min_length');
        set_error_notification($error);
    } else if (!empty($search_term)) {
        // 构建SQL查询
        $sql = "SELECT employee_id, username, fullname, role, profile_picture, created_at, updated_at FROM users WHERE ";

        // 根据选择的字段构建WHERE子句
        if ($search_field == 'employee_id' && is_numeric($search_term)) {
            $sql .= "employee_id = ?";
            $stmt = $conn->prepare($sql);
            $stmt->bind_param("i", $search_term);
        } elseif ($search_field == 'username') {
            $sql .= "username LIKE ?";
            $search_param = "%$search_term%";
            $stmt = $conn->prepare($sql);
            $stmt->bind_param("s", $search_param);
        } elseif ($search_field == 'fullname') {
            $sql .= "fullname LIKE ?";
            $search_param = "%$search_term%";
            $stmt = $conn->prepare($sql);
            $stmt->bind_param("s", $search_param);
        } elseif ($search_field == 'role') {
            $sql .= "role LIKE ?";
            $search_param = "%$search_term%";
            $stmt = $conn->prepare($sql);
            $stmt->bind_param("s", $search_param);
        } else {
            // 搜索所有字段
            $sql = "SELECT employee_id, username, fullname, role, profile_picture, created_at, updated_at FROM users WHERE
                    employee_id LIKE ? OR
                    username LIKE ? OR
                    fullname LIKE ? OR
                    role LIKE ?";
            $search_param = "%$search_term%";
            $stmt = $conn->prepare($sql);
            $stmt->bind_param("ssss", $search_param, $search_param, $search_param, $search_param);
        }

        // 执行查询
        $stmt->execute();
        $result = $stmt->get_result();

        // 获取结果
        if ($result && $result->num_rows > 0) {
            while($row = $result->fetch_assoc()) {
                $users[] = $row;
            }
            $total_results = count($users);
            $success = sprintf(__('showing_results') . " %d " . __('results') . " " . __('for') . " '%s'", $total_results, htmlspecialchars($search_term));
            set_success_notification($success);
        } else {
            $error = sprintf(__('no_results_found') . " '%s'", htmlspecialchars($search_term));
            set_error_notification($error);
        }

        $stmt->close();
    }
}

// 获取当前登录用户信息
$current_user_id = $_SESSION['id'];
$current_user_sql = "SELECT username, fullname, profile_picture FROM users WHERE employee_id = ?";
$stmt = $conn->prepare($current_user_sql);
$stmt->bind_param("i", $current_user_id);
$stmt->execute();
$current_user_result = $stmt->get_result();
$current_user = $current_user_result->fetch_assoc();
?>
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="icon.png" type="image/x-icon">
    <link rel="stylesheet" href="css/dashboard-style.css">
    <link rel="stylesheet" href="css/slidebar.css">
    <link rel="stylesheet" href="css/search-style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <?php
    // 语言文件已在文件开头加载
    include 'includes/theme-loader.php';
    include 'includes/date-formatter.php';
    include 'includes/notifications-loader.php';
    ?>
    <title><?php echo __('search'); ?> - FRAS System</title>
</head>
<body>
    <div class="dashboard-container">
        <?php include 'includes/sidebar.php'; ?>

        <!-- 主内容区 -->
        <div class="main-content">
            <!-- 顶部欢迎区 -->
            <div class="header">
                <div class="welcome-text">
                    <h1><?php echo __('welcome_back'); ?>, <?php echo htmlspecialchars($current_user['fullname']); ?></h1>
                    <p><?php echo __('today_is'); ?> <?php echo date('F j, Y'); ?></p>
                </div>

                <div class="header-actions">
                    <div class="user-profile">
                        <?php if (!empty($current_user['profile_picture'])): ?>
                            <img src="data:image/jpeg;base64,<?php echo base64_encode($current_user['profile_picture']); ?>" alt="Profile Picture">
                        <?php else: ?>
                            <img src="default_avatar.png" alt="Default Profile Picture">
                        <?php endif; ?>
                    </div>
                </div>
            </div>

            <!-- 内容标题区 -->
            <div class="content-header">
                <h2 class="content-title"><?php echo __('search'); ?></h2>

                <button onclick="window.location.href='dashboard.php'" class="action-button">
                    <i class="fas fa-arrow-left"></i>
                    <?php echo __('back_to_dashboard'); ?>
                </button>
            </div>

            <!-- 搜索区域 -->
            <div class="search-container">
                <form method="GET" class="search-form">
                    <div class="search-field">
                        <label for="search_term"><?php echo __('search_term'); ?>:</label>
                        <input type="text" id="search_term" name="search_term" value="<?php echo htmlspecialchars($search_term); ?>" placeholder="<?php echo __('search_term'); ?>...">
                    </div>

                    <div class="search-field">
                        <label for="search_field"><?php echo __('search_in'); ?>:</label>
                        <select id="search_field" name="search_field">
                            <option value="all" <?php echo $search_field == 'all' ? 'selected' : ''; ?>><?php echo __('all_fields'); ?></option>
                            <option value="employee_id" <?php echo $search_field == 'employee_id' ? 'selected' : ''; ?>><?php echo __('employee_id'); ?></option>
                            <option value="username" <?php echo $search_field == 'username' ? 'selected' : ''; ?>><?php echo __('username'); ?></option>
                            <option value="fullname" <?php echo $search_field == 'fullname' ? 'selected' : ''; ?>><?php echo __('full_name'); ?></option>
                            <option value="role" <?php echo $search_field == 'role' ? 'selected' : ''; ?>><?php echo __('role'); ?></option>
                        </select>
                    </div>

                    <div class="search-field" style="flex: 0 0 auto; display: flex; gap: 10px;">
                        <button type="submit" name="search" class="search-button">
                            <i class="fas fa-search"></i> <?php echo __('search'); ?>
                        </button>

                        <button type="button" onclick="window.location.href='search.php'" class="reset-button">
                            <i class="fas fa-redo"></i> <?php echo __('reset'); ?>
                        </button>
                    </div>
                </form>
            </div>

            <!-- 通知区域由notifications-loader.php处理 -->

            <!-- 搜索结果 -->
            <?php if (!empty($users)): ?>
                <div class="results-info">
                    <span><?php echo __('showing_results'); ?> <?php echo count($users); ?> <?php echo __('results'); ?></span>
                </div>

                <div class="data-table">
                    <table>
                        <thead>
                            <tr>
                                <th><?php echo __('id'); ?></th>
                                <th><?php echo __('avatar'); ?></th>
                                <th><?php echo __('username'); ?></th>
                                <th><?php echo __('full_name'); ?></th>
                                <th><?php echo __('role'); ?></th>
                                <th><?php echo __('created_at'); ?></th>
                                <th><?php echo __('updated_at'); ?></th>
                                <th><?php echo __('actions'); ?></th>
                            </tr>
                        </thead>
                        <tbody>
                            <?php foreach ($users as $row): ?>
                                <tr>
                                    <td><?php echo highlightSearchTerm($row['employee_id'], $search_term, $search_field); ?></td>
                                    <td>
                                        <div class="user-avatar">
                                            <?php if (!empty($row['profile_picture'])): ?>
                                                <img src="data:image/jpeg;base64,<?php echo base64_encode($row['profile_picture']); ?>" alt="Profile Picture">
                                            <?php else: ?>
                                                <img src="default_avatar.png" alt="Default Profile Picture">
                                            <?php endif; ?>
                                        </div>
                                    </td>
                                    <td><?php echo highlightSearchTerm($row['username'], $search_term, $search_field); ?></td>
                                    <td><?php echo highlightSearchTerm($row['fullname'], $search_term, $search_field); ?></td>
                                    <td>
                                        <span class="status-badge <?php echo $row['role'] == 'admin' ? 'status-pending' : 'status-open'; ?>">
                                            <?php echo highlightSearchTerm(($row['role'] == 'admin' ? 'Admin' : 'User'), $search_term, $search_field); ?>
                                        </span>
                                    </td>
                                    <td>
                                        <div class="date-time">
                                            <span class="date"><?php echo formatDate($row['created_at']); ?></span>
                                            <span class="time"><?php echo date('H:i', strtotime($row['created_at'])); ?></span>
                                        </div>
                                    </td>
                                    <td>
                                        <div class="date-time">
                                            <span class="date"><?php echo formatDate($row['updated_at']); ?></span>
                                            <span class="time"><?php echo date('H:i', strtotime($row['updated_at'])); ?></span>
                                        </div>
                                    </td>
                                    <td>
                                        <div class="action-buttons">
                                            <?php if ($_SESSION['role'] === 'admin'): ?>
                                            <!-- 管理员可以编辑和删除所有用户 -->
                                            <button onclick="window.location.href='edit.php?id=<?php echo htmlspecialchars($row['employee_id']); ?>'" class="edit-button">
                                                <i class="fas fa-edit"></i>
                                            </button>
                                            <button onclick="window.location.href='remove.php?id=<?php echo htmlspecialchars($row['employee_id']); ?>'" class="delete-button">
                                                <i class="fas fa-trash"></i>
                                            </button>
                                            <?php else: ?>
                                            <!-- 普通用户不能编辑或删除任何用户 -->
                                            <button class="edit-button disabled" title="<?php echo __('no_permission'); ?>" disabled>
                                                <i class="fas fa-lock"></i>
                                            </button>
                                            <button class="delete-button disabled" title="<?php echo __('no_permission'); ?>" disabled>
                                                <i class="fas fa-lock"></i>
                                            </button>
                                            <?php endif; ?>
                                        </div>
                                    </td>
                                </tr>
                            <?php endforeach; ?>
                        </tbody>
                    </table>
                </div>
            <?php elseif ($_SERVER["REQUEST_METHOD"] == "GET" && isset($_GET['search'])): ?>
                <div class="no-results">
                    <i class="fas fa-search"></i>
                    <p><?php echo __('no_results_found'); ?> "<?php echo htmlspecialchars($search_term); ?>"</p>
                    <p><?php echo __('try_different_search'); ?></p>
                </div>
            <?php endif; ?>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // 检测是否为移动设备
            const isMobile = window.innerWidth <= 768;

            if (isMobile) {
                // 获取所有表格容器
                const dataTables = document.querySelectorAll('.data-table');

                dataTables.forEach(table => {
                    // 添加滑动指示器
                    const indicator = document.createElement('div');
                    indicator.className = 'swipe-indicator';
                    indicator.innerHTML = '<i class="fas fa-arrows-alt-h"></i> ← Swipe to view more →';
                    indicator.style.cssText = 'text-align: center; padding: 10px; color: #777; font-size: 12px; background-color: #f9f9f9; border-top: 1px solid #eee; display: none;';

                    // 检查表格是否需要滚动
                    if (table.scrollWidth > table.clientWidth) {
                        indicator.style.display = 'block';
                        table.appendChild(indicator);

                        // 滚动时隐藏指示器
                        table.addEventListener('scroll', function() {
                            if (this.scrollLeft > 0) {
                                indicator.style.opacity = '0.3';
                            } else {
                                indicator.style.opacity = '1';
                            }
                        });
                    }
                });
            }
        });

        // 高亮搜索词的函数
        function highlightText(text, searchTerm) {
            if (!searchTerm || searchTerm.length < 2) return text;

            // 安全处理：转义正则表达式中的特殊字符
            const escapedSearchTerm = searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

            // 创建正则表达式并替换
            const regex = new RegExp(`(${escapedSearchTerm})`, 'gi');
            return text.replace(regex, '<span class="highlight">$1</span>');
        }
    </script>
</body>
</html>


