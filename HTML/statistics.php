<?php
session_start();
include 'config.php';
include 'includes/language-loader.php';

// Check if user is logged in
if (!isset($_SESSION['loggedin']) || $_SESSION['loggedin'] !== true) {
    header("Location: login.php");
    exit();
}

// Get current user information
$current_user_id = $_SESSION['id'];
$current_user_sql = "SELECT username, fullname, profile_picture, role FROM users WHERE employee_id = ?";
$stmt = $conn->prepare($current_user_sql);
$stmt->bind_param("i", $current_user_id);
$stmt->execute();
$current_user_result = $stmt->get_result();
$current_user = $current_user_result->fetch_assoc();

// Get filter parameters
$selected_employee = isset($_GET['employee_id']) ? intval($_GET['employee_id']) : $current_user_id;
$selected_month = isset($_GET['month']) ? intval($_GET['month']) : date('n');
$selected_year = isset($_GET['year']) ? intval($_GET['year']) : date('Y');

// Check permissions - only admins can view other employees' statistics
if ($selected_employee != $current_user_id && $current_user['role'] !== 'admin') {
    $selected_employee = $current_user_id;
}

// Get employee list for admin users
$employees = [];
if ($current_user['role'] === 'admin') {
    // Get all users with their details for better selection
    $employees_sql = "
        SELECT
            u.employee_id,
            u.username,
            u.fullname,
            u.role,
            u.created_at,
            (SELECT COUNT(*) FROM attendance a WHERE a.employee_id = u.employee_id) as attendance_count
        FROM
            users u
        ORDER BY
            u.fullname
    ";
    $employees_result = $conn->query($employees_sql);
    while ($row = $employees_result->fetch_assoc()) {
        $employees[] = $row;
    }
}

// Get attendance summary for the selected month and year
$start_date = "$selected_year-$selected_month-01 00:00:00";
$end_date = date('Y-m-t 23:59:59', strtotime($start_date));

// Calculate working days in the month
$days_in_month = cal_days_in_month(CAL_GREGORIAN, $selected_month, $selected_year);
$working_days = 0;
$future_days = 0;
$current_day = date('j');
$current_month = date('n');
$current_year = date('Y');

for ($day = 1; $day <= $days_in_month; $day++) {
    $date = mktime(0, 0, 0, $selected_month, $day, $selected_year);
    $weekday = date('N', $date); // 1 (Monday) to 7 (Sunday)

    // Check if it's a working day (Monday to Saturday)
    if ($weekday < 7) {
        $working_days++;

        // Check if this day is in the future - only for current month or future months
        if (($selected_year > $current_year) ||
            ($selected_year == $current_year && $selected_month > $current_month) ||
            ($selected_year == $current_year && $selected_month == $current_month && $day > $current_day)) {
            $future_days++;
        }

        // 如果是过去的月份，确保没有未来的日子
        if ($selected_year < $current_year ||
            ($selected_year == $current_year && $selected_month < $current_month)) {
            $future_days = 0; // 过去的月份没有未来的日子
        }
    }
}

// Get work time settings
$work_time_sql = "SELECT start_time, end_time, created_at FROM work_time_settings ORDER BY created_at DESC LIMIT 1";
$work_time_result = $conn->query($work_time_sql);

// Default values
$start_time = '08:30:00';
$end_time = '18:00:00';

// If we have work time settings, use them
if ($work_time_result && $work_time_result->num_rows > 0) {
    $work_time = $work_time_result->fetch_assoc();
    $start_time = $work_time['start_time'];
    $end_time = $work_time['end_time'];
    $last_updated = $work_time['created_at'];
} else {
    $last_updated = 'Default values used';
}

// Debug information
$work_time_debug = [
    'start_time' => $start_time,
    'end_time' => $end_time,
    'last_updated' => $last_updated
];

// Get detailed attendance data for each day in the month
$detailed_attendance_sql = "
    SELECT
        DATE(a_in.created_at) as attendance_date,
        MIN(TIME(a_in.created_at)) as check_in_time,
        MAX(TIME(a_out.created_at)) as check_out_time
    FROM
        attendance a_in
    LEFT JOIN
        attendance a_out ON DATE(a_in.created_at) = DATE(a_out.created_at)
        AND a_out.employee_id = a_in.employee_id
        AND a_out.session_type = 'out'
    WHERE
        a_in.employee_id = ?
        AND a_in.created_at BETWEEN ? AND ?
        AND a_in.session_type = 'in'
    GROUP BY
        DATE(a_in.created_at)
";
$stmt = $conn->prepare($detailed_attendance_sql);
$stmt->bind_param("iss", $selected_employee, $start_date, $end_date);
$stmt->execute();
$detailed_result = $stmt->get_result();

// Initialize counters
$days_attended = 0;
$days_on_time = 0;
$days_full_time = 0;
$total_check_ins = 0;

// Process each day's attendance
$daily_attendance = [];
while ($row = $detailed_result->fetch_assoc()) {
    $days_attended++;
    $total_check_ins++;

    $check_in_time = $row['check_in_time'];
    $check_out_time = $row['check_out_time'];
    $attendance_date = $row['attendance_date'];
    $day = date('j', strtotime($attendance_date));

    // Store for daily attendance display
    $daily_attendance[$day] = $check_in_time;

    // Check if on time (check-in before or at start time)
    if ($check_in_time <= $start_time) {
        $days_on_time++;
    }

    // Check if full time (check-in before or at start time AND check-out after or at end time)
    if ($check_in_time <= $start_time && $check_out_time >= $end_time) {
        $days_full_time++;
    }
}

// Calculate attendance rates
$on_time_rate = ($days_attended > 0) ? round(($days_on_time / $days_attended) * 100, 2) : 0;
$full_time_rate = ($days_attended > 0) ? round(($days_full_time / $days_attended) * 100, 2) : 0;
$attendance_rate = ($working_days > 0) ? round(($days_attended / $working_days) * 100, 2) : 0;

// Generate monthly trend data
$monthly_trend = [];
$months = [];
$rates = [];
$attendance_counts = [];
$workday_counts = [];

for ($month = 1; $month <= 12; $month++) {
    $month_start = date('Y-m-01', mktime(0, 0, 0, $month, 1, $selected_year));
    $month_end = date('Y-m-t', mktime(0, 0, 0, $month, 1, $selected_year));

    // Get detailed attendance data for this month
    $month_sql = "
        SELECT
            DATE(a_in.created_at) as attendance_date,
            MIN(TIME(a_in.created_at)) as check_in_time,
            MAX(TIME(a_out.created_at)) as check_out_time
        FROM
            attendance a_in
        LEFT JOIN
            attendance a_out ON DATE(a_in.created_at) = DATE(a_out.created_at)
            AND a_out.employee_id = a_in.employee_id
            AND a_out.session_type = 'out'
        WHERE
            a_in.employee_id = ?
            AND a_in.created_at BETWEEN ? AND ?
            AND a_in.session_type = 'in'
        GROUP BY
            DATE(a_in.created_at)
    ";

    $month_stmt = $conn->prepare($month_sql);
    $month_stmt->bind_param("iss", $selected_employee, $month_start, $month_end);
    $month_stmt->execute();
    $month_result = $month_stmt->get_result();

    // Initialize counters for this month
    $days_attended_month = 0;
    $days_on_time_month = 0;
    $days_full_time_month = 0;

    // Process each day's attendance for this month
    while ($row = $month_result->fetch_assoc()) {
        $days_attended_month++;

        $check_in_time = $row['check_in_time'];
        $check_out_time = $row['check_out_time'];

        // Check if on time (check-in before or at start time)
        if ($check_in_time <= $start_time) {
            $days_on_time_month++;
        }

        // Check if full time (check-in before or at start time AND check-out after or at end time)
        if ($check_in_time <= $start_time && $check_out_time >= $end_time) {
            $days_full_time_month++;
        }
    }

    // Calculate working days in this month
    $days_in_month = date('t', mktime(0, 0, 0, $month, 1, $selected_year));
    $working_days_month = 0;

    for ($day = 1; $day <= $days_in_month; $day++) {
        $date = mktime(0, 0, 0, $month, $day, $selected_year);
        $weekday = date('N', $date); // 1 (Monday) to 7 (Sunday)
        if ($weekday < 7) { // Monday to Saturday
            $working_days_month++;
        }
    }

    // Calculate rates
    $attendance_rate_month = ($working_days_month > 0) ? round(($days_attended_month / $working_days_month) * 100, 2) : 0;
    $on_time_rate_month = ($days_attended_month > 0) ? round(($days_on_time_month / $days_attended_month) * 100, 2) : 0;
    $full_time_rate_month = ($days_attended_month > 0) ? round(($days_full_time_month / $days_attended_month) * 100, 2) : 0;

    // Add to arrays
    $months[] = date('M', mktime(0, 0, 0, $month, 1, $selected_year)); // Month abbreviation
    $attendance_rates[] = $attendance_rate_month;
    $on_time_rates[] = $on_time_rate_month;
    $full_time_rates[] = $full_time_rate_month;
    $attendance_counts[] = $days_attended_month;
    $on_time_counts[] = $days_on_time_month;
    $full_time_counts[] = $days_full_time_month;
    $workday_counts[] = $working_days_month;

    // Don't process future months
    if ($selected_year == date('Y') && $month > date('n')) {
        break;
    }
}

// Convert to JSON for JavaScript
$monthly_trend_json = json_encode([
    'months' => $months,
    'attendance_rates' => $attendance_rates,
    'on_time_rates' => $on_time_rates,
    'full_time_rates' => $full_time_rates,
    'attendance_counts' => $attendance_counts,
    'on_time_counts' => $on_time_counts,
    'full_time_counts' => $full_time_counts,
    'workday_counts' => $workday_counts
]);

// Get daily attendance data for the selected month
$daily_attendance_sql = "
    SELECT
        DAY(created_at) as day,
        MIN(TIME(created_at)) as check_in_time
    FROM
        attendance
    WHERE
        employee_id = ?
        AND created_at BETWEEN ? AND ?
        AND session_type = 'in'
    GROUP BY
        DAY(created_at)
    ORDER BY
        day
";
$stmt = $conn->prepare($daily_attendance_sql);
$stmt->bind_param("iss", $selected_employee, $start_date, $end_date);
$stmt->execute();
$daily_attendance_result = $stmt->get_result();
$daily_attendance = [];
while ($row = $daily_attendance_result->fetch_assoc()) {
    $daily_attendance[$row['day']] = $row['check_in_time'];
}

// Get selected employee name
$employee_name = $current_user['fullname'];
if ($selected_employee != $current_user_id) {
    $emp_sql = "SELECT fullname FROM users WHERE employee_id = ?";
    $stmt = $conn->prepare($emp_sql);
    $stmt->bind_param("i", $selected_employee);
    $stmt->execute();
    $emp_result = $stmt->get_result();
    if ($emp_row = $emp_result->fetch_assoc()) {
        $employee_name = $emp_row['fullname'];
    }
}
?>
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="icon.png" type="image/x-icon">
    <link rel="stylesheet" href="css/dashboard-style.css">
    <link rel="stylesheet" href="css/slidebar.css">
    <link rel="stylesheet" href="css/statistics-style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.css">
    <?php
    include 'includes/theme-loader.php';
    include 'includes/date-formatter.php';
    ?>
    <title><?php echo __('statistics'); ?> - FRAS System</title>
</head>
<body>
    <div class="dashboard-container">
        <?php include 'includes/sidebar.php'; ?>

        <!-- Main Content -->
        <div class="main-content">
            <!-- Header -->
            <div class="content-header">
                <h2 class="content-title"><?php echo __('attendance_statistics'); ?></h2>

                <div class="user-profile">
                    <img src="data:image/jpeg;base64,<?php echo base64_encode($current_user['profile_picture']); ?>" alt="Profile Picture">
                </div>
            </div>

            <!-- Filter Section -->
            <div class="filter-section">
                <form action="" method="GET" class="filter-form">
                    <div class="filter-group employee-filter-group">
                        <label for="employee_id"><?php echo __('employee'); ?>:</label>
                        <?php if ($current_user['role'] !== 'admin'): ?>
                            <input type="hidden" name="employee_id" value="<?php echo $current_user_id; ?>">
                            <div class="employee-display">
                                <div class="employee-info">
                                    <div class="employee-name"><?php echo $current_user['fullname']; ?></div>
                                    <div class="employee-details"><?php echo __('id'); ?>: <?php echo $current_user_id; ?></div>
                                </div>
                            </div>
                        <?php else: ?>
                            <div class="employee-select-container">
                                <div class="employee-search">
                                    <input type="text" id="employeeSearch" placeholder="<?php echo __('search_employees'); ?>" autocomplete="off">
                                    <i class="fas fa-search search-icon"></i>
                                </div>
                                <select name="employee_id" id="employee_id" class="employee-select">
                                    <?php foreach ($employees as $employee): ?>
                                        <option value="<?php echo $employee['employee_id']; ?>"
                                                <?php echo ($selected_employee == $employee['employee_id']) ? 'selected' : ''; ?>
                                                data-username="<?php echo $employee['username']; ?>"
                                                data-role="<?php echo $employee['role']; ?>"
                                                data-attendance="<?php echo $employee['attendance_count']; ?>">
                                            <?php echo $employee['fullname']; ?> (<?php echo $employee['employee_id']; ?>)
                                        </option>
                                    <?php endforeach; ?>
                                </select>
                                <div class="selected-employee-details">
                                    <div class="detail-item">
                                        <span class="detail-label"><?php echo __('username'); ?>:</span>
                                        <span class="detail-value" id="selectedUsername"></span>
                                    </div>
                                    <div class="detail-item">
                                        <span class="detail-label"><?php echo __('role'); ?>:</span>
                                        <span class="detail-value" id="selectedRole"></span>
                                    </div>
                                    <div class="detail-item">
                                        <span class="detail-label"><?php echo __('total_attendance_records'); ?>:</span>
                                        <span class="detail-value" id="selectedAttendance"></span>
                                    </div>
                                </div>
                            </div>
                        <?php endif; ?>
                    </div>

                    <div class="filter-group">
                        <label for="month"><?php echo __('month'); ?>:</label>
                        <select name="month" id="month">
                            <?php
                            $month_names = [
                                1 => __('January'), 2 => __('February'), 3 => __('March'), 4 => __('April'),
                                5 => __('May'), 6 => __('June'), 7 => __('July'), 8 => __('August'),
                                9 => __('September'), 10 => __('October'), 11 => __('November'), 12 => __('December')
                            ];

                            foreach ($month_names as $num => $name) {
                                echo '<option value="' . $num . '" ' . ($selected_month == $num ? 'selected' : '') . '>' . $name . '</option>';
                            }
                            ?>
                        </select>
                    </div>

                    <div class="filter-group">
                        <label for="year"><?php echo __('year'); ?>:</label>
                        <select name="year" id="year">
                            <?php
                            $current_year = date('Y');
                            for ($year = $current_year; $year >= $current_year - 5; $year--) {
                                echo '<option value="' . $year . '" ' . ($selected_year == $year ? 'selected' : '') . '>' . $year . '</option>';
                            }
                            ?>
                        </select>
                    </div>

                    <button type="submit" class="filter-button">
                        <i class="fas fa-filter"></i> <?php echo __('filter'); ?>
                    </button>
                </form>
            </div>



            <!-- Statistics Cards -->
            <div class="statistics-cards">
                <div class="stat-card">
                    <div class="stat-icon">
                        <i class="fas fa-calendar-check"></i>
                    </div>
                    <div class="stat-info">
                        <h3><?php echo __('days_attended'); ?></h3>
                        <p class="stat-value">
                            <span class="attendance-numbers"><?php echo $days_attended; ?>/<?php echo $working_days; ?></span>
                            <?php if ($future_days > 0): ?>
                                <span class="future-days" title="<?php echo __('future'); ?>">(<?php echo $future_days; ?> <?php echo __('future'); ?>)</span>
                            <?php endif; ?>
                        </p>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-icon">
                        <i class="fas fa-clock"></i>
                    </div>
                    <div class="stat-info">
                        <h3><?php echo __('on_time_rate'); ?></h3>
                        <p class="stat-value"><?php echo $on_time_rate; ?>%</p>
                        <p class="stat-subtitle"><?php echo __('before'); ?> <?php echo date('H:i', strtotime($start_time)); ?></p>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-icon">
                        <i class="fas fa-business-time"></i>
                    </div>
                    <div class="stat-info">
                        <h3><?php echo __('full_time_rate'); ?></h3>
                        <p class="stat-value"><?php echo $full_time_rate; ?>%</p>
                        <p class="stat-subtitle"><?php echo date('H:i', strtotime($start_time)); ?> - <?php echo date('H:i', strtotime($end_time)); ?></p>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-icon">
                        <i class="fas fa-percentage"></i>
                    </div>
                    <div class="stat-info">
                        <h3><?php echo __('attendance_rate'); ?></h3>
                        <p class="stat-value"><?php echo $attendance_rate; ?>%</p>
                    </div>
                </div>
            </div>

            <!-- Charts Section -->
            <div class="charts-section">
                <!-- Attendance Pie Chart -->
                <div class="chart-container">
                    <h3 class="chart-title"><?php echo __('attendance_distribution'); ?></h3>
                    <div class="chart-wrapper">
                        <canvas id="attendancePieChart"></canvas>
                    </div>
                </div>



                <!-- Monthly Trend Chart -->
                <div class="chart-container">
                    <h3 class="chart-title"><?php echo __('monthly_trend'); ?></h3>
                    <div class="chart-wrapper">
                        <canvas id="monthlyTrendChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js"></script>
    <script>
        // 移除 datalabels 插件注册，可能导致图表不显示
        // Chart.register(ChartDataLabels);

        // Attendance Chart with Rose Colors
        const attendanceCtx = document.getElementById('attendancePieChart').getContext('2d');

        // Define the attendance data
        const attendanceData = {
            onTime: <?php echo $days_on_time; ?>,
            late: <?php echo $days_attended - $days_on_time; ?>,
            absent: <?php echo $working_days - $days_attended - $future_days; ?>,
            future: <?php echo $future_days; ?>
        };

        // 检查是否有未来的日子
        const hasFutureDays = <?php echo $future_days > 0 ? 'true' : 'false'; ?>;

        // Create the chart - using polarArea type with rose colors
        const attendancePieChart = new Chart(attendanceCtx, {
            type: 'polarArea',
            data: {
                // 根据是否有未来的日子动态设置标签
                labels: hasFutureDays ? [
                    '<?php echo __('on_time'); ?>',
                    '<?php echo __('late'); ?>',
                    '<?php echo __('absent'); ?>',
                    '<?php echo __('future'); ?>'
                ] : [
                    '<?php echo __('on_time'); ?>',
                    '<?php echo __('late'); ?>',
                    '<?php echo __('absent'); ?>'
                ],
                datasets: [{
                    // 根据是否有未来的日子动态设置数据
                    data: hasFutureDays ? [
                        // 使用原始数据，饼图会自动根据比例调整扇区大小
                        attendanceData.onTime,
                        attendanceData.late,
                        attendanceData.absent,
                        attendanceData.future
                    ] : [
                        attendanceData.onTime,
                        attendanceData.late,
                        attendanceData.absent
                    ],
                    backgroundColor: hasFutureDays ? [
                        'rgba(221, 160, 221, 0.7)',  // Lavender rose (purple) for on time
                        'rgba(255, 182, 193, 0.7)',  // Pink rose for late
                        'rgba(178, 34, 34, 0.7)',    // Deep red rose for absent
                        'rgba(255, 250, 205, 0.7)'   // Light yellow rose for future
                    ] : [
                        'rgba(221, 160, 221, 0.7)',  // Lavender rose (purple) for on time
                        'rgba(255, 182, 193, 0.7)',  // Pink rose for late
                        'rgba(178, 34, 34, 0.7)'     // Deep red rose for absent
                    ],
                    borderColor: hasFutureDays ? [
                        'rgba(221, 160, 221, 1)',    // Lavender rose (purple)
                        'rgba(255, 182, 193, 1)',    // Pink rose
                        'rgba(178, 34, 34, 1)',      // Deep red rose
                        'rgba(255, 250, 205, 1)'     // Light yellow rose
                    ] : [
                        'rgba(221, 160, 221, 1)',    // Lavender rose (purple)
                        'rgba(255, 182, 193, 1)',    // Pink rose
                        'rgba(178, 34, 34, 1)'       // Deep red rose
                    ],
                    borderWidth: 2,
                    hoverBackgroundColor: hasFutureDays ? [
                        'rgba(221, 160, 221, 0.9)',  // Lavender rose (purple)
                        'rgba(255, 182, 193, 0.9)',  // Pink rose
                        'rgba(178, 34, 34, 0.9)',    // Deep red rose
                        'rgba(255, 250, 205, 0.9)'   // Light yellow rose
                    ] : [
                        'rgba(221, 160, 221, 0.9)',  // Lavender rose (purple)
                        'rgba(255, 182, 193, 0.9)',  // Pink rose
                        'rgba(178, 34, 34, 0.9)'     // Deep red rose
                    ],
                    hoverBorderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                // 修复图表被截断的问题
                radius: '70%', // 减小半径，使图表更紧凑
                layout: {
                    padding: {
                        top: 20,
                        right: 20,
                        bottom: 20,
                        left: 20
                    }
                },
                scales: {
                    r: {
                        ticks: {
                            display: false
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        // 确保图表完全显示
                        afterFit: function(scaleInstance) {
                            scaleInstance.height = scaleInstance.height * 0.9;
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            font: {
                                size: 16
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;

                                // 计算饼图的总和作为分母
                                const chartData = context.chart.data.datasets[0].data;
                                const sum = chartData.reduce((a, b) => a + b, 0);

                                // 使用饼图总和计算百分比，确保总和为100%
                                const percentage = sum > 0 ? Math.round((value / sum) * 100) : 0;

                                // 显示工作日总数
                                const workingDays = <?php echo $working_days; ?>;

                                // Add work time information to tooltip
                                if (label === '<?php echo __('on_time'); ?>') {
                                    return `${label}: ${value}/${workingDays} (${percentage}%) - <?php echo __('before'); ?> <?php echo date('H:i', strtotime($start_time)); ?>`;
                                } else if (label === '<?php echo __('late'); ?>') {
                                    return `${label}: ${value}/${workingDays} (${percentage}%) - <?php echo __('after'); ?> <?php echo date('H:i', strtotime($start_time)); ?>`;
                                } else if (label === '<?php echo __('absent'); ?>') {
                                    return `${label}: ${value}/${workingDays} (${percentage}%)`;
                                } else if (label === '<?php echo __('future'); ?>') {
                                    return `${label}: ${value}/${workingDays} (${percentage}%)`;
                                } else {
                                    return `${label}: ${value}/${workingDays} (${percentage}%)`;
                                }
                            }
                        }
                    }
                },
                animation: {
                    duration: 3000,
                    easing: 'easeInOutQuart' // Slow-fast-slow rhythm
                },
                onClick: function(event, elements) {
                    if (elements.length > 0) {
                        const index = elements[0].index;
                        showAttendanceDetails(index);
                    }
                }
            }
        });

        // Function to show attendance details in a modal
        function showAttendanceDetails(index) {
            // Get the category and value - 根据是否有未来的日子动态设置类别
            const categories = hasFutureDays ?
                ['on_time', 'late', 'absent', 'future'] :
                ['on_time', 'late', 'absent'];
            const category = categories[index];
            const value = hasFutureDays ? [
                attendanceData.onTime,
                attendanceData.late,
                attendanceData.absent,
                attendanceData.future
            ][index] : [
                attendanceData.onTime,
                attendanceData.late,
                attendanceData.absent
            ][index];

            // Rose colors for modal headers - 根据是否有未来的日子动态设置颜色
            const roseColors = hasFutureDays ? [
                '#DDA0DD', // Lavender rose (purple) for on time
                '#FFB6C1', // Pink rose for late
                '#B22222', // Deep red rose for absent
                '#FFFACD'  // Light yellow rose for future
            ] : [
                '#DDA0DD', // Lavender rose (purple) for on time
                '#FFB6C1', // Pink rose for late
                '#B22222'  // Deep red rose for absent
            ];

            // Create or get the modal
            let modal = document.getElementById('attendanceDetailsModal');
            if (!modal) {
                modal = document.createElement('div');
                modal.id = 'attendanceDetailsModal';
                modal.className = 'attendance-modal';
                document.body.appendChild(modal);
            }

            // Set the content based on the category
            let title = '';
            let content = '';
            let total, percentage;
            let headerColor = roseColors[index];

            // 计算饼图的总和作为分母，确保百分比总和为100%
            const chartSum = hasFutureDays ?
                (attendanceData.onTime + attendanceData.late + attendanceData.absent + attendanceData.future) :
                (attendanceData.onTime + attendanceData.late + attendanceData.absent);

            // 使用饼图总和计算百分比
            percentage = chartSum > 0 ? Math.round((value / chartSum) * 100) : 0;

            // 获取工作日总数，用于显示
            const workingDays = <?php echo $working_days; ?>;

            if (category === 'on_time') {
                title = __('on_time');
                content = `
                    <div class="modal-stat">
                        <span class="stat-label">${__('days')}:</span>
                        <span class="stat-value">${value}</span>
                    </div>
                    <div class="modal-stat">
                        <span class="stat-label">${__('percentage')}:</span>
                        <span class="stat-value">${percentage}% (${value}/${workingDays} ${__('days')})</span>
                    </div>
                    <div class="modal-stat">
                        <span class="stat-label">${__('Work start time')}:</span>
                        <span class="stat-value"><?php echo date('H:i', strtotime($start_time)); ?></span>
                    </div>
                    <p class="modal-description">
                        ${__('on_time_description')}
                    </p>
                `;
            } else if (category === 'late') {
                title = __('late');
                content = `
                    <div class="modal-stat">
                        <span class="stat-label">${__('days')}:</span>
                        <span class="stat-value">${value}</span>
                    </div>
                    <div class="modal-stat">
                        <span class="stat-label">${__('percentage')}:</span>
                        <span class="stat-value">${percentage}% (${value}/${workingDays} ${__('days')})</span>
                    </div>
                    <div class="modal-stat">
                        <span class="stat-label">${__('Work start time')}:</span>
                        <span class="stat-value"><?php echo date('H:i', strtotime($start_time)); ?></span>
                    </div>
                    <p class="modal-description">
                        ${__('late_description')}
                    </p>
                `;
            } else if (category === 'absent') {
                title = __('absent');
                content = `
                    <div class="modal-stat">
                        <span class="stat-label">${__('days')}:</span>
                        <span class="stat-value">${value}</span>
                    </div>
                    <div class="modal-stat">
                        <span class="stat-label">${__('percentage')}:</span>
                        <span class="stat-value">${percentage}% (${value}/${workingDays} ${__('days')})</span>
                    </div>
                    <p class="modal-description">
                        ${__('absent_description')}
                    </p>
                `;
            } else if (category === 'future') {
                title = __('future');
                content = `
                    <div class="modal-stat">
                        <span class="stat-label">${__('days')}:</span>
                        <span class="stat-value">${value}</span>
                    </div>
                    <div class="modal-stat">
                        <span class="stat-label">${__('percentage')}:</span>
                        <span class="stat-value">${percentage}% (${value}/${workingDays} ${__('days')})</span>
                    </div>
                    <p class="modal-description">
                        ${__('future_description')}
                    </p>
                `;
            }

            // 设置模态框内容，使用玫瑰色样式并增加视觉效果
            modal.innerHTML = `
                <div class="modal-content">
                    <div class="modal-header" style="background-color: ${headerColor}; color: ${category === 'future' ? '#333' : '#fff'};">
                        <h2>${title}</h2>
                        <button class="close-button" onclick="closeAttendanceModal()" style="color: ${category === 'future' ? '#333' : '#fff'};">&times;</button>
                    </div>
                    <div class="modal-body">
                        <div class="modal-icon" style="background-color: ${headerColor}; opacity: 0.2; border-radius: 50%; width: 80px; height: 80px; position: absolute; right: 20px; top: 70px; z-index: 0;"></div>
                        <div style="position: relative; z-index: 1;">
                            ${content}
                        </div>
                    </div>
                </div>
            `;

            // Show the modal
            modal.style.display = 'flex';
        }

        // Function to close the modal
        window.closeAttendanceModal = function() {
            const modal = document.getElementById('attendanceDetailsModal');
            if (modal) {
                modal.style.display = 'none';
            }
        };



        // Monthly Trend Chart - Direct Data with Rose Colors
        const monthlyData = <?php echo $monthly_trend_json; ?>;
        const monthlyCtx = document.getElementById('monthlyTrendChart').getContext('2d');
        const monthlyTrendChart = new Chart(monthlyCtx, {
            type: 'line',
            data: {
                labels: monthlyData.months,
                datasets: [
                    {
                        label: '<?php echo __('attendance_rate'); ?> (%)',
                        data: monthlyData.attendance_rates,
                        backgroundColor: 'rgba(178, 34, 34, 0.1)',    // Deep red rose
                        borderColor: 'rgba(178, 34, 34, 1)',          // Deep red rose
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: '<?php echo __('on_time_rate'); ?> (%)',
                        data: monthlyData.on_time_rates,
                        backgroundColor: 'rgba(221, 160, 221, 0.1)',  // Lavender rose
                        borderColor: 'rgba(221, 160, 221, 1)',        // Lavender rose
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: '<?php echo __('full_time_rate'); ?> (%)',
                        data: monthlyData.full_time_rates,
                        backgroundColor: 'rgba(255, 182, 193, 0.1)',  // Pink rose
                        borderColor: 'rgba(255, 182, 193, 1)',        // Pink rose
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const index = context.dataIndex;
                                const datasetIndex = context.datasetIndex;
                                const workdays = monthlyData.workday_counts[index];

                                if (datasetIndex === 0) {
                                    // Attendance rate
                                    const rate = monthlyData.attendance_rates[index];
                                    const attended = monthlyData.attendance_counts[index];
                                    return [
                                        `${__('attendance_rate')}: ${rate}%`,
                                        `${__('days_attended')}: ${attended}/${workdays}`
                                    ];
                                } else if (datasetIndex === 1) {
                                    // On-time rate
                                    const rate = monthlyData.on_time_rates[index];
                                    const onTime = monthlyData.on_time_counts[index];
                                    const attended = monthlyData.attendance_counts[index];
                                    return [
                                        `${__('on_time_rate')}: ${rate}%`,
                                        `${__('on_time')}: ${onTime}/${attended} ${__('days')}`,
                                        `${__('before')}: <?php echo date('H:i', strtotime($start_time)); ?>`
                                    ];
                                } else if (datasetIndex === 2) {
                                    // Full-time rate
                                    const rate = monthlyData.full_time_rates[index];
                                    const fullTime = monthlyData.full_time_counts[index];
                                    const attended = monthlyData.attendance_counts[index];
                                    return [
                                        `${__('full_time_rate')}: ${rate}%`,
                                        `${__('full_time')}: ${fullTime}/${attended} ${__('days')}`,
                                        `${__('work_hours')}: <?php echo date('H:i', strtotime($start_time)); ?> - <?php echo date('H:i', strtotime($end_time)); ?>`
                                    ];
                                }
                            }
                        }
                    }
                }
            }
        });

        // Helper function for translations in JavaScript
        function __(key) {
            const translations = {
                'attendance_rate': '<?php echo __('attendance_rate'); ?>',
                'days_attended': '<?php echo __('days_attended'); ?>',
                'weekend': '<?php echo __('weekend'); ?>',
                'check_in_time': '<?php echo __('check_in_time'); ?>',
                'admin': '<?php echo __('admin'); ?>',
                'user': '<?php echo __('user'); ?>',
                'present': '<?php echo __('present'); ?>',
                'absent': '<?php echo __('absent'); ?>',
                'future': '<?php echo __('future'); ?>',
                'check_ins': '<?php echo __('check_ins'); ?>',
                'attendance_distribution': '<?php echo __('attendance_distribution'); ?>',
                'on_time_rate': '<?php echo __('on_time_rate'); ?>',
                'full_time_rate': '<?php echo __('full_time_rate'); ?>',
                'on_time': '<?php echo __('on_time'); ?>',
                'late': '<?php echo __('late'); ?>',
                'full_time': '<?php echo __('full_time'); ?>',
                'before': '<?php echo __('before'); ?>',
                'after': '<?php echo __('after'); ?>',
                'work_hours': '<?php echo __('work_hours'); ?>',
                'days': '<?php echo __('days'); ?>',
                'percentage': '<?php echo __('percentage'); ?>',
                'of_working_days': '<?php echo __('of_working_days'); ?>',
                'of_attended_days': '<?php echo __('of_attended_days'); ?>',
                'on_time_description': '<?php echo __('on_time_description'); ?>',
                'late_description': '<?php echo __('late_description'); ?>',
                'absent_description': '<?php echo __('absent_description'); ?>',
                'future_description': '<?php echo __('future_description'); ?>'
            };
            return translations[key] || key;
        }
    </script>



    <!-- Employee Selection Script -->
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Employee search functionality
            const employeeSearch = document.getElementById('employeeSearch');
            const employeeSelect = document.getElementById('employee_id');

            if (employeeSearch && employeeSelect) {
                // Initialize employee details display
                updateEmployeeDetails();

                // Search functionality
                employeeSearch.addEventListener('input', function() {
                    const searchTerm = this.value.toLowerCase();
                    const options = employeeSelect.options;

                    for (let i = 0; i < options.length; i++) {
                        const optionText = options[i].text.toLowerCase();
                        const optionUsername = options[i].getAttribute('data-username').toLowerCase();

                        if (optionText.includes(searchTerm) || optionUsername.includes(searchTerm)) {
                            options[i].style.display = '';
                        } else {
                            options[i].style.display = 'none';
                        }
                    }
                });

                // Update details when selection changes
                employeeSelect.addEventListener('change', updateEmployeeDetails);

                function updateEmployeeDetails() {
                    const selectedOption = employeeSelect.options[employeeSelect.selectedIndex];

                    if (selectedOption) {
                        const username = selectedOption.getAttribute('data-username');
                        const role = selectedOption.getAttribute('data-role');
                        const attendance = selectedOption.getAttribute('data-attendance');

                        document.getElementById('selectedUsername').textContent = username;
                        document.getElementById('selectedRole').textContent = role === 'admin' ? __('admin') : __('user');
                        document.getElementById('selectedAttendance').textContent = attendance;
                    }
                }
            }
        });
    </script>
</body>
</html>